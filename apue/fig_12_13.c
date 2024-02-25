#include "apue.h"

extern char** environ;

static pthread_key_t key;
static pthread_mutex_t env_mutex;
static pthread_once_t init_done = PTHREAD_ONCE_INIT;

static void thread_init()
{
	pthread_key_create(&key, free);
}

char* getenv(const char* name)
{
	pthread_once(&init_done, thread_init);

	pthread_mutex_lock(&env_mutex);
	char* envbuf = (char*)pthread_getspecific(key);
	if(envbuf == NULL)
	{
		envbuf = (char*)malloc(BUFSIZ);
		if(envbuf == NULL)
		{
			pthread_mutex_unlock(&env_mutex);
			return NULL;
		}
		pthread_setspecific(key, envbuf);
	}

	size_t len = strlen(name);
	for(int i=0; environ[i] != NULL; i++)
	{
		if(strncmp(name, environ[i], len) == 0 
			&& environ[i][len] == '=')
		{
			strncpy(envbuf, &environ[i][len+1], BUFSIZ-1);
			pthread_mutex_unlock(&env_mutex);
			return envbuf;
		}
	}
	pthread_mutex_unlock(&env_mutex);
	return NULL;
}

