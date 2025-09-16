"""
One of FEMR's main features is utilities for helping write labeling functions.

The following are two simple labelers for inpatient mortality and long admission for MIMIC-IV.
"""

import femr.labelers
import meds_reader
import meds
import datetime
import shutil

from typing import List, Mapping
from pathlib import Path


LABEL_NAMES = [
    "death",
    #"long_los",
]
# LABEL_NAMES = ['long_los', '30d']
# ADMISSION_EVENTS = ["Visit/IP", "Visit/ERIP", "Visit/ER", "CMS Place of Service/51", "CMS Place of Service/61"]
# MEDS transforms omop etl
ADMISSION_EVENTS = ["Visit//IP//start", "Visit//ERIP//start", "Visit//ER//start",
                    "CMS Place of Service//20//start", "CMS Place of Service//15//start"]
DISCHARGE_EVENTS = ["Visit//IP//end", "Visit//ERIP//end", "Visit//ER//end",
                    "CMS Place of Service//20//end", "CMS Place of Service//15//end"]
ADMISSION_EVENTS_OUTPATIENT = ["Visit//OP//start", "CMS Place of Service//22//start"]
END_TIMES_INCLUDED=False
VERBOSE=False

def collect_stays(subject, end_times_included=False, verbose=False):
    admission_ranges = set()
    death_times = set()
    latest_admission = None
    if not end_times_included:
        admission_dict = {}
        for event in subject.events:
            if event.code in ADMISSION_EVENTS:
                admission_dict[event.time] = event
                latest_admission = event
            if event.code in DISCHARGE_EVENTS:
                possible_admissions = [t for t in admission_dict if t < event.time]
                if possible_admissions:
                    latest_time = max(possible_admissions)
                    latest_admission = admission_dict.pop(latest_time)
                    admission_ranges.add((latest_admission.time, event.time))
                    if verbose:
                        print(
                            f"Matched admission for subject {subject.subject_id} from {latest_admission.time} to {event.time} "
                            f"with admission event {latest_admission.code} and discharge event {event.code}")
                        # print(
                        #     f"Found admission for subject {subject.subject_id} from {latest_admission.time} to {event.time} "
                        #     f"with admission event {latest_admission.code} and discharge event {event.code}")
            if event.code == meds.death_code:
                death_times.add(event.time)
                # if verbose:
                # print(f"Found death for subject {subject.subject_id} at {event.time} time since admission "
                #       f"{event.time - max(admission_dict.keys())}")
    else:
        for event in subject.events:
            if event.code in ADMISSION_EVENTS and event.end is not None:
                if isinstance(event.end, datetime.datetime):
                    admission_ranges.add((event.time, event.end))
                else:
                    admission_ranges.add((event.time, datetime.datetime.fromisoformat(event.end)))
            if event.code == meds.death_code:
                death_times.add(event.time)

    return admission_ranges, death_times

class OmopInpatientMortalityLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta):
        self.time_after_admission = time_after_admission

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        labels = []
        admission_ranges, death_times = collect_stays(subject, END_TIMES_INCLUDED, VERBOSE)
        if len(death_times) not in [0, 1]:
            print(f"Warning: found {len(death_times)} death events in subject: {subject.subject_id}")

        if len(death_times) == 1:
            death_time = list(death_times)[0]
        else:
            # death_time = datetime.datetime(9999, 1, 1)  # Very far in the future
            for (admission_start, admission_end) in admission_ranges:
                prediction_time = admission_start + self.time_after_admission
                labels.append(
                    meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time, boolean_value=False))
            return labels
        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            # if prediction_time >= admission_end:
            #     continue

            if prediction_time >= death_time:
                print(f"Warning: prediction time {prediction_time} is after death time {death_time} for subject {subject.subject_id}")
                continue
            if death_time > admission_end:
                print(f"Warning: death time {death_time} is before prediction time {prediction_time} for subject {subject.subject_id}")
                # continue
            is_death = death_time < prediction_time #< admission_end
            if is_death:
                print(f"Labeling subject {subject.subject_id} as death at prediction time {prediction_time} with death time {death_time}")
            labels.append(
                meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time, boolean_value=is_death))

        return labels




class OmopLongAdmissionLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta, admission_length: datetime.timedelta):
        self.time_after_admission = time_after_admission
        self.admission_length = admission_length

    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        admission_ranges = set()

        # for event in subject.events:
        #     if event.code in ADMISSION_EVENTS and event.end is not None:
        #         if isinstance(event.end, datetime.datetime):
        #             admission_ranges.add((event.time, event.end))
        #         else:
        #             admission_ranges.add((event.time, datetime.datetime.fromisoformat(event.end)))
        collect_stays(subject, end_times_included=END_TIMES_INCLUDED, verbose=VERBOSE)
        labels = []
        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue

            is_long_admission = (admission_end - admission_start) > self.admission_length

            labels.append(meds.Label(subject_id=subject.subject_id, prediction_time=prediction_time,
                                     boolean_value=is_long_admission))

        return labels


labelers: Mapping[str, femr.labelers.Labeler] = {
    'death': OmopInpatientMortalityLabeler(time_after_admission=datetime.timedelta(hours=48)),
    'long_los': OmopLongAdmissionLabeler(time_after_admission=datetime.timedelta(hours=48),
                                         admission_length=datetime.timedelta(days=7)),
}


def create_omop_meds_tutorial_arg_parser():
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
    parser.add_argument("--num_threads", dest="num_threads", type=int, default=6)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=False)
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
    parser.add_argument("--end_times_included", dest="end_times_included", action="store_true", default=False)
    return parser


def main():
    args = create_omop_meds_tutorial_arg_parser().parse_args()
    if args.verbose:
        global VERBOSE
        VERBOSE = True
        print("Verbose logging enabled")
    if args.end_times_included:
        global END_TIMES_INCLUDED
        END_TIMES_INCLUDED = True
        print("Assuming end times are included in the data (event.end)")
    labels_path = Path(args.pretraining_data) / "labels"
    if labels_path.exists() and not args.overwrite:
        raise ValueError(f"Labels path {labels_path} already exists. Use --overwrite to overwrite.")
    labels_path.mkdir(exist_ok=True, parents=True)

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=args.num_threads) as database:
        for label_name in LABEL_NAMES:
            print(f"Labeling {label_name}")
            labeler = labelers[label_name]
            labels = labeler.apply(database)
            labels.to_parquet(str(labels_path / (label_name + '.parquet')))


if __name__ == "__main__":
    main()
