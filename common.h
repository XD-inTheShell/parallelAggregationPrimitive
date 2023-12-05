#define KEYSIZE 128
#define VALUEINT
#ifdef VALUEINT
    using Value = uint32_t;
#else
    using Value = double;
#endif
using Key = uint32_t;