#include "cli_serv.h"

int read_stream(int fd, char* data, int maxBytes)
{
	int bytesLeft, bytesRead; 
	bytesLeft = maxBytes;
	while(bytesLeft > 0)
	{
		if ((bytesRead = read(fd, data, bytesLeft)) < 0)
			return bytesRead;
		else if(bytesRead == 0)
			break; /*EOF*/
		bytesLeft -= bytesRead;
		data += bytesRead;
	}

	return (maxBytes - bytesLeft);
}
