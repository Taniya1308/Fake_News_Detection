"""
Generates a small demo dataset (data/Fake.csv + data/True.csv)
so the pipeline can be run without downloading the full Kaggle dataset.

The sample contains 200 fake + 200 real articles (synthetic but representative).
Run:  python src/generate_sample_data.py
"""

import os
import random
import pandas as pd

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Fake news templates ────────────────────────────────────────────────────────
FAKE_TITLES = [
    "SHOCKING: {topic} EXPOSED by insider leak!!!",
    "You won't BELIEVE what {person} just admitted about {topic}",
    "BREAKING: Government hides truth about {topic} — wake up sheeple!",
    "Secret documents PROVE {person} controls {topic}",
    "{topic} is a HOAX designed to control the population",
    "Scientists BAFFLED by {topic} — mainstream media silent",
    "EXCLUSIVE: {person} caught lying about {topic} on live TV",
    "The {topic} agenda: what they don't want you to know",
]
FAKE_BODIES = [
    "Sources close to the situation confirmed that {topic} has been manipulated "
    "for decades. {person} was seen at a secret meeting last week. Share before "
    "they delete this!!!",
    "A whistleblower has come forward with damning evidence that {topic} is "
    "entirely fabricated. {person} denies involvement but leaked emails say "
    "otherwise. The truth is out there.",
    "Independent researchers have uncovered a massive cover-up involving {topic}. "
    "{person} is at the center of the conspiracy. Big media refuses to report "
    "this story.",
]
REAL_TITLES = [
    "Study finds new evidence linking {topic} to public health outcomes",
    "{person} announces policy changes on {topic} at press conference",
    "Experts weigh in on the latest developments in {topic}",
    "Report: {topic} shows significant improvement over previous quarter",
    "{person} and world leaders meet to discuss {topic} challenges",
    "Research team publishes findings on the impact of {topic}",
    "Government releases data on {topic} amid growing concerns",
    "Analysis: How {topic} is shaping the economy this year",
]
REAL_BODIES = [
    "According to a peer-reviewed study published in the journal Nature, {topic} "
    "has measurable effects on communities worldwide. {person} commented that "
    "further research is needed to fully understand the implications.",
    "At a press conference on Monday, {person} outlined new initiatives aimed at "
    "addressing {topic}. Officials say the policy is backed by extensive data "
    "collected over the past five years.",
    "Data released by the Department of Statistics shows that {topic} indicators "
    "have improved by 12% compared to last year. {person} called the results "
    "encouraging while cautioning against complacency.",
]

TOPICS  = ["climate change", "vaccines", "election results", "economic policy",
           "immigration", "healthcare reform", "artificial intelligence", "space exploration"]
PERSONS = ["President Smith", "Senator Johnson", "Dr. Williams", "CEO Brown",
           "Mayor Davis", "Governor Wilson", "Secretary Lee", "Director Taylor"]


def make_rows(titles, bodies, n=200):
    rows = []
    for _ in range(n):
        title = random.choice(titles).format(
            topic=random.choice(TOPICS), person=random.choice(PERSONS)
        )
        text = random.choice(bodies).format(
            topic=random.choice(TOPICS), person=random.choice(PERSONS)
        )
        rows.append({"title": title, "text": text, "subject": "news", "date": "2023-01-01"})
    return rows


fake_df = pd.DataFrame(make_rows(FAKE_TITLES, FAKE_BODIES, 200))
true_df = pd.DataFrame(make_rows(REAL_TITLES, REAL_BODIES, 200))

fake_path = os.path.join(DATA_DIR, "Fake.csv")
true_path = os.path.join(DATA_DIR, "True.csv")

fake_df.to_csv(fake_path, index=False)
true_df.to_csv(true_path, index=False)

print(f"Sample dataset created:")
print(f"  {fake_path}  ({len(fake_df)} rows)")
print(f"  {true_path}  ({len(true_df)} rows)")
print("\nYou can now run:  python src/train.py")
