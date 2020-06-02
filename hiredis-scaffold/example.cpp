#include <iostream>
#include <string>
#include <memory>

#include <hiredis/hiredis.h>

using namespace std;

typedef std::shared_ptr<redisReply> RedisReplySharedPtr;

class RedisConnectionCpp
{
public:
    RedisConnectionCpp():m_conn(NULL) {}
    ~RedisConnectionCpp() { redisFree(m_conn); }

    // port will be ignored if use_unix_socket is enabled
    bool initConnection(bool use_unix_socket, const char* host, int port, const timeval& timeout);

    RedisReplySharedPtr executeCommand(const char* cmd, ...);

private:
    static void RedisReply_cleaner(redisReply* p) { freeReplyObject(p); }

private:
    redisContext* m_conn;
};

bool RedisConnectionCpp::initConnection(bool use_unix_socket, const char* host, int port, const timeval& timeout)
{
    if (use_unix_socket) {
        m_conn = redisConnectUnixWithTimeout(host, timeout);
    } else {
        m_conn = redisConnectWithTimeout(host, port, timeout);
    }
    
    if (m_conn == NULL || m_conn->err) {
        if (m_conn) {
            cout << "Connection error: " <<  m_conn->errstr << endl;
        } else {
            cout << "Connection error: can't allocate redis context" << endl;
        }
        return false;
    }
    return true;
}

RedisReplySharedPtr RedisConnectionCpp::executeCommand(const char* cmd, ...)
{
    va_list args;
    va_start(args, cmd);
    redisReply* p = (redisReply*)redisvCommand(m_conn, cmd, args);
    va_end(args);
    return RedisReplySharedPtr(p, RedisReply_cleaner);
}

int main(int argc, char* argv[]) 
{
    const char* hostname = (argc > 1) ? argv[1] : "127.0.0.1";

    bool isunix = false;
    if (argc > 2) {
        if (*argv[2] == 'u' || *argv[2] == 'U') {
            isunix = true;
            printf("Will connect to unix socket @%s\n", hostname);
        }
    }

    int port = (argc > 2) ? atoi(argv[2]) : 6379;
    struct timeval timeout = { 1, 500000 }; // 1.5 seconds

    RedisConnectionCpp conn;
    if(!conn.initConnection(isunix, hostname, port, timeout)) {
        exit(EXIT_FAILURE);
    }

    /* PING server */
    auto reply = conn.executeCommand("PING");
    printf("PING: %s\n", reply->str);

    /* Set a key */
    reply = conn.executeCommand("SET %s %s", "foo", "hello world");
    printf("SET: %s\n", reply->str);

    /* Set a key using binary safe API */
    reply = conn.executeCommand("SET %b %b", "bar", (size_t) 3, "hello", (size_t) 5);
    printf("SET (binary API): %s\n", reply->str);

    /* Try a GET and two INCR */
    reply = conn.executeCommand("GET foo");
    printf("GET foo: %s\n", reply->str);

    reply = conn.executeCommand("INCR counter");
    printf("INCR counter: %lld\n", reply->integer);

    /* again ... */
    reply = conn.executeCommand("INCR counter");
    printf("INCR counter: %lld\n", reply->integer);

    /* Create a list of numbers, from 0 to 9 */
    reply = conn.executeCommand("DEL mylist");
    for (int j = 0; j < 10; j++)
    {
        char buf[64];
        snprintf(buf,64,"%u",j);
        conn.executeCommand("LPUSH mylist element-%s", buf);
    }

    /* Let's check what we have inside the list */
    reply = conn.executeCommand("LRANGE mylist 0 -1");
    if (reply->type == REDIS_REPLY_ARRAY) 
    {
        for (int j = 0; j < reply->elements; j++)
        {
            printf("%u) %s\n", j, reply->element[j]->str);
        }
    }

    return 0;
}
