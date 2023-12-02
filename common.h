#define KEYSIZE 50
#define VALUEINT
#ifdef VALUEINT
    using Value = uint32_t;
#else
    using Value = double;
#endif