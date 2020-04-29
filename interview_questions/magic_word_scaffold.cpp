#include "leetcode.h"
#include "util/trie_tree.h"

using namespace std;
using namespace osrm;

/* leetcode exercises: 676 */

int main()
{
    util::LogPolicy::GetInstance().Unmute();

    util::Log(logESSENTIAL) << "Running replaceWords tests:";
    TIMER_START(replaceWords);
    replaceWords_scaffold("[cat,bat,rat]", "the cattle was rattled by the battery", "the cat was rat by the bat");
    TIMER_STOP(replaceWords);
    util::Log(logESSENTIAL) << "replaceWords using " << TIMER_MSEC(replaceWords) << " milliseconds";


}
