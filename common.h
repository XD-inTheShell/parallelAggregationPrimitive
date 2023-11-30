#define KEYSIZE 50
#define VALUEINT
#ifdef VALUEINT
    using Value = int;
#else
    using Value = double;
#endif