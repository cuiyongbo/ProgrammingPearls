#include "apue.h"

struct Job
{
	pthread_t j_id;
	struct Job* j_prev;
	struct Job* j_next;
};


struct Queue
{
	struct Job* q_head;
	struct Job* q_tail;
	pthread_rwlock_t q_lock;
};

int queue_init(struct Queue* qp)
{
	qp->q_head = qp->q_tail = NULL;
	int err = pthread_rwlock_init(&qp->q_lock, NULL);
	if(err != 0)
		return err;

	// continue initialization
	return 0;
}

// insert a Job* at the head of the queue
void job_insert(struct Queue* qp, struct Job* jp)
{
	pthread_rwlock_wrlock(&qp->q_lock);
	jp->j_next = qp->q_head;
	jp->j_prev = NULL;
	if(qp->q_head != NULL)
		qp->q_head->j_prev = jp;
	else
		qp->q_tail = jp;
	qp->q_head = jp;
	pthread_rwlock_unlock(&qp->q_lock);
}

// append a Job* on the tail of the queue
void job_append(struct Queue* qp, struct Job* jp)
{
	pthread_rwlock_wrlock(&qp->q_lock);
	jp->j_next = NULL;
	jp->j_prev = qp->q_tail;
	if(qp->q_tail != NULL)
		qp->q_tail->j_next = jp;
	else
		qp->q_head = jp;
	qp->q_tail = jp;
	pthread_rwlock_unlock(&qp->q_lock);
}

void job_remove(struct Queue* qp, struct Job* jp)
{
	pthread_rwlock_wrlock(&qp->q_lock);
	if(jp == qp->q_head)
	{
		qp->q_head = jp->j_next;
		if(qp->q_tail == jp)
			qp->q_tail = NULL;
		else
			jp->j_next->j_prev = jp->j_prev;
	}
	else if(jp == qp->q_tail)
	{
		qp->q_tail = jp->j_prev;
		jp->j_next->j_prev = jp->j_next;
	}	
	else
	{
		jp->j_prev->j_next = jp->j_next;
		jp->j_next->j_prev = jp->j_prev;
	}
	pthread_rwlock_unlock(&qp->q_lock);
}

struct Job* job_find(struct Queue* qp, pthread_t id)
{
	struct Job* jp;
	if(pthread_rwlock_rdlock(&qp->q_lock) != 0)
		return NULL;

	for(jp=qp->q_head; jp != NULL; jp=jp->j_next)
	{
		if(pthread_equal(jp->j_id, id))
			break;
	}
	pthread_rwlock_unlock(&qp->q_lock);
	return jp;
}

