from celery import Celery

def make_celery():
    celery = Celery(
        'test',
        broker='redis://localhost:6379/0',
        backend='redis://localhost:6379/0'
    )
    return celery

celery = make_celery()
