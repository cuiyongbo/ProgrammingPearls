#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int isMatchedSuffix(const char* word, const char* suffix, size_t suffixLen);

int main(int argc, char* argv[])
{
	if(argc != 2) {
		printf("Usage: %s word\n", argv[0]);
		return 1;
	}

	char suffixes[] = "etic|alistic|stic|ptic|lytic|otic|antic|ntic|ctic|atic|hnic|nic|mic|llic|blic|clic|lic|hic|fic|dic|bic|aic|mac|iac";
	const char* sep = "|";
	char* next;
	char* cur = strtok_r(suffixes, sep, &next); // suffixes is modified
	
	int found = 0;
	while(cur) {
			//printf("cur: %s, next: %s\n", cur, next);
			found = isMatchedSuffix(argv[1], cur, next-cur-1);
			if(found == 1)
				break;
			cur = strtok_r(NULL, sep, &next);
	}

	if(found == 1) {
		char suffix[8];
		memcpy(suffix, cur, next-cur);
		printf("<%s>'s suffix: %s\n", argv[1], suffix);
	} else {
		printf("<%s>'s suffix not found!\n", argv[1]);
	}

	return 0;
}

int isMatchedSuffix(const char* word, const char* suffix, size_t suffixLen)
{
	size_t wordLen = strlen(word);
	
	if(wordLen < suffixLen)
		return 0;
	else if(wordLen == suffixLen)
		return strcmp(word, suffix) == 0;
	else
		return strcmp(word + wordLen-suffixLen, suffix) == 0;	
}
