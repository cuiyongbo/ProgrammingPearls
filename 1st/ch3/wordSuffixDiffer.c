#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* suffixes = "etic|alistic|stic|ptic|lytic|otic|antic|ntic|ctic|atic|\
						hnic|nic|mic|llic|blic|clic|lic|hic|fic|dic|bic|aic|mac|iac";


int isMatchedSuffix(const char* word, const char* suffix)
{
	size_t suffixLen = strlen(suffix);
	size_t wordLen = strlen(word);
	
	if(wordLen < suffixLen)
		return 0;
	else if(wordLen == suffixLen)
		return strcmp(word, suffix) == 0;
	else
		return strcmp(word + wordLen-suffixLen, suffix) == 0;	
}

int main(int argc, char* argv[])
{
	if(argc != 2) {
		printf("Usage: %s word\n", argv[0]);
		return 0;
	}

	return 0;
}

