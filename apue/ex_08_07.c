#include "apue.h"
#include <dirent.h>

int main()
{
    DIR* dirp = opendir(".");
    if(dirp == NULL)
        err_sys("opendir error");

    struct dirent* entry = NULL;
    for (entry = readdir(dirp); entry != NULL; entry = readdir(dirp))
    {
        printf("%s, type: %d\n", entry->d_name, entry->d_type);
    }

    printf("close-on-exec flag: %d\n", fcntl(dirfd(dirp), F_GETFD));
    //closedir(dirp);

    int fd = open(".", O_RDONLY);
    if (fd < 0)
        err_sys("open error");

    printf("close-on-exec flag: %d\n", fcntl(fd, F_GETFD));
    return 0;
}


int fcntl(int fd, int cmd, ... /* arg */ );
