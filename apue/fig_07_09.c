#include "apue.h"
#include <setjmp.h>

#define TOK_ADD 5

void do_line(char*);
void cmd_add();
int get_token();

int main()
{
	char line[MAXLINE];
	
	while(fgets(line, MAXLINE, stdin) != NULL)
		do_line(line);
	exit(0);
}

char* tok_ptr; /* global pointer for get_token() */

void do_line(char* ptr)
{
	int cmd;
	tok_ptr = ptr;	
	while((cmd = get_token()) > 0)
	{
		switch(cmd)
		{
			case TOK_ADD:
				cmd_add();
				break;
		}
	}
}

void cmd_add()
{
	int token = get_token();
	/* rest of processing for this command*/
}

int get_token()
{
	/* fetch next token from line pointed by tok_ptr */
}


