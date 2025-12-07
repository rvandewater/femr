import femr.ontology
import pathlib
import meds_reader
import pickle
import femr.splits
from femr.models.tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
import femr.models.tasks
import femr.models.processor
import pandas as pd
import polars as pl
from timeit import default_timer as timer

def main(args):
    pretraining_data_path = pathlib.Path(args.pretraining_data)
    meds_reader_path = pathlib.Path(args.meds_reader)
    subject_splits_path = meds_reader_path / "metadata/subject_splits.parquet"
    code_metadata_path = meds_reader_path / "metadata/codes.parquet"
    codes_to_skip = []
    num_threads = args.num_threads
    if args.motor_codes_to_skip:
        codes_to_skip = pl.read_parquet(args.motor_codes_to_skip)
        codes_to_skip = codes_to_skip["code"].to_list()


    with meds_reader.SubjectDatabase(str(meds_reader_path), num_threads=num_threads) as database:
        subject_ids = [_ for _ in database]
        ontology_path = pretraining_data_path / 'ontology.pkl'
        if not ontology_path.exists():
            print("Creating ontology")
            ontology = femr.ontology.Ontology(args.athena_path, code_metadata_path=str(code_metadata_path))
            print("Pruning the ontology")
            ontology.prune_to_dataset(
                database,
                prune_all_descriptions=True,
                remove_ontologies={'SPL', 'HemOnc', 'LOINC'}
            )

            with open(ontology_path, 'wb') as f:
                pickle.dump(ontology, f)
        else:
            with open(ontology_path, 'rb') as f:
                ontology = pickle.load(f)
        if not args.pre_split:
subject_splits_path = "/sc/arion/projects/hpims-hpi/projects/foundation_models_ehr/cohorts/meds_debug/full_omop_25_04_29/MEDS_cohort/metadata/subject_splits.parquet"
subject_splits = pl.read_parquet(subject_splits_path)
train_val_ids = subject_splits.filter(pl.col("split") != "held_out").select("subject_id").to_series().to_list()
train_ids = subject_splits.filter(pl.col("split") == "train").select("subject_id").to_series().to_list()
test_split = subject_splits.filter(pl.col("split") == "held_out").select("subject_id").to_series().to_list()
main_split = femr.splits.SubjectSplit(train_val_ids, test_split)
main_split.save_to_csv("/sc/arion/projects/hpims-hpi/projects/foundation_models_ehr/meds_data/25-08-01_meds_etl_base/motor_data/motor_model/" + 'main_split.csv')

            train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.05)
            train_val_ids = main_split.train_subject_ids
            train_ids = train_split.train_subject_ids
            val_ids = train_split.test_subject_ids
        else:
subject_splits = pl.read_parquet(subject_splits_path)
train_val_ids = subject_splits.filter(pl.col("split") != "held_out").select("subject_id").to_series().to_list()
train_ids = subject_splits.filter(pl.col("split") == "train").select("subject_id").to_series().to_list()
test_split = subject_splits.filter(pl.col("split") == "held_out").select("subject_id").to_series().to_list()
val_ids = subject_splits.filter(pl.col("split") == "tuning").select("subject_id").to_series().to_list()
        main_database = database.filter(train_val_ids)
        train_database = main_database.filter(train_ids)
        val_database = main_database.filter(val_ids)
        print(f"Training with {len(train_database)} training subjects and {len(val_database)} validation subjects")
        tokenizer_path = pretraining_data_path / 'tokenizer'
        if not tokenizer_path.exists():
            print("Train tokenizer. This will take several hours")
            tokenizer_start = timer()
            tokenizer = HierarchicalTokenizer.train(
                main_database,
                vocab_size=1024 * 16,
                ontology=ontology,
            )
            # Save the tokenizer to the same directory as the model
            tokenizer.save_pretrained(tokenizer_path)
            tokenizer_end = timer()
            print(f"Tokenizer trained in {tokenizer_end - tokenizer_start / 3600:.2f} hours")
        else:
            tokenizer = HierarchicalTokenizer.from_pretrained(tokenizer_path, ontology=ontology)

        task_path = pretraining_data_path / 'motor_task.pkl'

        if not task_path.exists():
            # Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit
            print("Train MOTOR task. This will take several hours")
            task_start = timer()
            motor_task = femr.models.tasks.MOTORTask.fit_pretraining_task_info(
                main_database, tokenizer,
                num_tasks=16384, #8 * 1024,
                num_bins=8,
                final_layer_size=512,
                codes_to_skip=codes_to_skip
            )
            task_end = timer()
            print(f"MOTOR task trained in {task_end - task_start / 3600:.2f} hours")
            with open(task_path, 'wb') as f:
                pickle.dump(motor_task, f)

        else:
            with open(task_path, 'rb') as f:
                motor_task = pickle.load(f)

        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

        example_subject_id = list(train_database)[0]
        example_subject = train_database[example_subject_id]

        # We can do this one subject at a time
        print("Convert a single subject")
        example_batch = processor.collate([processor.convert_subject(example_subject, tensor_type='pt')])

        train_batches_path = pretraining_data_path / 'train_batches'

        if not train_batches_path.exists():
            print("Convert batches")
            # But generally we want to convert entire datasets
            train_batches = processor.convert_dataset(
                train_database,
                tokens_per_batch=args.tokens_per_batch,
                min_subjects_per_batch=1,
                num_proc=num_threads
            )

            print("Convert batches to pytorch")
            # Convert our batches to pytorch tensors
            train_batches.set_format("pt")
            train_batches.save_to_disk(train_batches_path)

        val_batches_path = pretraining_data_path / 'val_batches'

        if not val_batches_path.exists():
            print("Convert val batches")
            val_batches = processor.convert_dataset(val_database, tokens_per_batch=args.tokens_per_batch, num_proc=num_threads)
            # Convert our batches to pytorch tensors
            val_batches.set_format("pt")
            val_batches.save_to_disk(val_batches_path)


def create_omop_meds_tutorial_argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for preparing Motor")
    parser.add_argument(
        "--pretraining_data",
        dest="pretraining_data",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--meds_reader",
        dest="meds_reader",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--athena_path",
        dest="athena_path",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--motor_codes_to_skip",
        dest="motor_codes_to_skip",
        action="store",
        required=False,
    )
    parser.add_argument(
        "--num_threads",
        dest="num_threads",
        action="store",
        required=False,
        type=int,
        default=16,
    )
    parser.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        action="store",
        required=False,
        type=int,
        # this is decided based on the 99% percentile of the number of tokens
        default=16384,
    )
    parser.add_argument("--pre_split",
                        action="store",
                        dest="pre_split",
                        default=False,
                        required=False,
                        help="Whether the splits have been generated already in train, tuning, and held_out sets")
    return parser


if __name__ == "__main__":
    main(create_omop_meds_tutorial_argparser().parse_args())
