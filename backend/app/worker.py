import os

from celery import Celery

celery_app = Celery("rag", broker=os.environ["CELERY_BROKER_URL"], include=["app.tasks"])

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    # Status lives in the documents table, so no result backend is needed.
    task_ignore_result=True,
    # Ack only after the task finishes: if the worker dies mid-task, RabbitMQ hands the message out again.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Take one message at a time, so a waiting document is not stuck behind a busy worker.
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # RabbitMQ 4 rejects transient non-exclusive queues, which is how Celery declares its
    # worker-control queue by default; the worker then drops its connection in a loop.
    control_queue_exclusive=True,
    # Same rule for the event queues that worker gossip uses.
    event_queue_exclusive=True,
    # An upload waits on the publish, so give up fast when RabbitMQ is down instead of hanging.
    broker_connection_timeout=3,
    task_publish_retry_policy={"max_retries": 2, "interval_start": 0, "interval_step": 1, "interval_max": 1},
)
