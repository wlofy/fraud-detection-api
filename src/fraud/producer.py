"""Replay transactions onto a Kafka topic to simulate a live stream.

    python -m fraud.producer            # replay held-out test split
    python -m fraud.producer --rate 200 # 200 txns/sec

Each message is a JSON transaction (Time, V1..V28, Amount) plus the true
``Class`` label so the consumer can report live precision/recall.
"""

from __future__ import annotations

import argparse
import json
import time

from kafka import KafkaProducer

from .config import FEATURE_COLUMNS, LABEL_COLUMN, settings
from .data import load_split


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay transactions to Kafka.")
    parser.add_argument("--rate", type=float, default=settings.producer_rate,
                        help="transactions/sec (0 = max speed)")
    parser.add_argument("--limit", type=int, default=0,
                        help="max messages to send (0 = entire test split)")
    args = parser.parse_args()

    _, test_df, _ = load_split()
    if args.limit:
        test_df = test_df.head(args.limit)

    producer = build_producer()
    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    cols = FEATURE_COLUMNS + [LABEL_COLUMN]

    print(f"Streaming {len(test_df):,} transactions to "
          f"'{settings.kafka_topic}' @ {args.rate or 'max'} tps …")
    sent = 0
    t0 = time.time()
    for txn_id, row in enumerate(test_df[cols].itertuples(index=False)):
        payload = dict(zip(cols, row))
        payload["txn_id"] = txn_id
        producer.send(settings.kafka_topic, payload)
        sent += 1
        if delay:
            time.sleep(delay)
        if sent % 1000 == 0:
            print(f"  sent {sent:,}")

    producer.flush()
    elapsed = time.time() - t0
    print(f"Done: {sent:,} messages in {elapsed:.1f}s "
          f"({sent / elapsed:.0f} tps effective).")


if __name__ == "__main__":
    main()
