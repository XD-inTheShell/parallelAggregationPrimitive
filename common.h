#define KEYSIZE 100
#define VALUEINT
#ifdef VALUEINT
    using Value = unsigned int;
#else
    using Value = double;
#endif
using Key = unsigned int;