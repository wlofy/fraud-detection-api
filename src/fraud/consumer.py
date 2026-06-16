"""Consume the Kafka transaction stream and score each one in real time.

    python -m fraud.consumer
    python -m fraud.consumer --model autoencoder

Prints a line for every flagged transaction and a rolling precision/recall
summary (possible because the producer ships the true label alongside).
"""

from __future__ import annotations

import argparse
import json

from kafka import KafkaConsumer

from .config import LABEL_COLUMN, settings
from .scoring import Scorer


class LiveStats:
    def __init__(self) -> None:
        self.n = self.tp = self.fp = self.fn = self.tn = 0

    def update(self, label: int | None, flagged: bool) -> None:
        self.n += 1
        if label is None:
            return
        if flagged and label == 1:
            self.tp += 1
        elif flagged and label == 0:
            self.fp += 1
        elif not flagged and label == 1:
            self.fn += 1
        else:
            self.tn += 1

    def summary(self) -> str:
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return (f"n={self.n:,} flagged={self.tp + self.fp} "
                f"TP={self.tp} FP={self.fp} FN={self.fn} "
                f"| precision={prec:.3f} recall={rec:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the Kafka stream.")
    parser.add_argument("--model", default=None, help="override scoring model")
    parser.add_argument("--from-beginning", action="store_true",
                        help="read the topic from the earliest offset")
    parser.add_argument("--idle-timeout", type=float, default=0.0,
                        help="exit after N seconds with no new messages "
                             "(0 = run forever)")
    parser.add_argument("--group", default="fraud-scorer",
                        help="Kafka consumer group id")
    args = parser.parse_args()

    scorer = Scorer(args.model)
    print(f"Scoring '{settings.kafka_topic}' with '{scorer.model_name}' "
          f"(threshold={scorer.threshold:.4f}). Ctrl-C to stop.\n")

    consumer = KafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest" if args.from_beginning else "latest",
        group_id=args.group,
        enable_auto_commit=True,
        consumer_timeout_ms=int(args.idle_timeout * 1000) if args.idle_timeout else None,
    )

    stats = LiveStats()
    try:
        for msg in consumer:
            txn = dict(msg.value)
            label = txn.pop(LABEL_COLUMN, None)
            txn_id = txn.pop("txn_id", None)
            result = scorer.score(txn)
            stats.update(label, result["is_fraud"])

            if result["is_fraud"]:
                tag = ""
                if label is not None:
                    tag = "  [TP]" if label == 1 else "  [FP]"
                print(f"FLAG txn#{txn_id} risk={result['risk_score']:.4f}"
                      f" amount={txn.get('Amount'):.2f}{tag}")

            if stats.n % 2000 == 0:
                print(f"  [{stats.summary()}]")
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nFinal: {stats.summary()}")
        consumer.close()


if __name__ == "__main__":
    main()
