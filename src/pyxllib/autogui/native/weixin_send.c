#include <windows.h>
#include <stdint.h>
#include <string.h>

/*
 * 版本无关的进程内微信文本发送适配器。
 *
 * 函数地址与消息结构偏移全部由调用方（weixin4_offsets 解析结果）通过
 * SendParams 传入，因此微信更新后不需要重新编译本适配器；只需重新解析布局。
 */

typedef void (*fn_one)(void *out);
typedef void (*fn_two)(uint64_t value, void *out);
typedef intptr_t (*fn_ctor)(void *object);
typedef intptr_t (*fn_send)(uint64_t context, void *scratch, void *vector, int flag);

typedef struct {
    uintptr_t get_coro;
    uintptr_t get_service;
    uintptr_t get_context;
    uintptr_t message_ctor;
    uintptr_t do_send;
    uint32_t object_size;
    uint32_t content_off;
    uint32_t wxid_off;
    uint32_t len_off;
    uint32_t flag_off;
    uint32_t kind_off;
    uint32_t subtype_off;
    uint32_t flag_value;
    uint32_t kind_value;
    uint32_t subtype_value;
} SendParams;

static void noop(void *value) { (void)value; }
static void *fake_vtable[2] = {(void *)&noop, (void *)&noop};

static void assign_string(unsigned char *field, const char *text) {
    size_t length = strlen(text);
    memset(field, 0, 32);
    *(uint64_t *)(field + 16) = (uint64_t)length;
    if (length <= 15) {
        memcpy(field, text, length);
        field[length] = 0;
        *(uint64_t *)(field + 24) = 15;
        return;
    }
    size_t capacity = length | 15;
    char *buffer = (char *)HeapAlloc(GetProcessHeap(), 0, capacity + 1);
    if (!buffer) return;
    memcpy(buffer, text, length);
    buffer[length] = 0;
    *(void **)field = buffer;
    *(uint64_t *)(field + 24) = (uint64_t)capacity;
}

static void release_shared(uint64_t control_value) {
    if (!control_value) return;
    unsigned char *control = (unsigned char *)(uintptr_t)control_value;
    volatile LONG *uses = (volatile LONG *)(control + 8);
    volatile LONG *weaks = (volatile LONG *)(control + 12);
    if (InterlockedDecrement(uses) == 0) {
        void **vtable = *(void ***)control;
        ((void (*)(void *))vtable[0])(control);
        if (InterlockedDecrement(weaks) == 0)
            ((void (*)(void *))vtable[1])(control);
    }
}

__declspec(dllexport) int SendTextNow(
    const SendParams *params, const char *wxid, const char *content,
    uintptr_t *diagnostics
) {
    if (!params) return 1;

    fn_one get_coro = (fn_one)params->get_coro;
    fn_two get_service = (fn_two)params->get_service;
    fn_two get_context = (fn_two)params->get_context;
    fn_ctor message_ctor = (fn_ctor)params->message_ctor;
    fn_send do_send = (fn_send)params->do_send;

    const size_t object_size = params->object_size;
    unsigned char *full = (unsigned char *)HeapAlloc(
        GetProcessHeap(), HEAP_ZERO_MEMORY, 0x10 + object_size + 64);
    if (!full) return 0;
    unsigned char *control = full;
    unsigned char *object = full + 0x10;
    unsigned char *element = object + object_size;
    unsigned char *vector = element + 16;

    *(void ***)control = fake_vtable;
    *(LONG *)(control + 8) = 10;
    *(LONG *)(control + 12) = 1;
    message_ctor(object);
    *(void **)(object + 8) = object;
    *(void **)(object + 16) = control;
    *(uint32_t *)(object + params->kind_off) = params->kind_value;
    *(uint32_t *)(object + params->flag_off) = params->flag_value;
    *(uint64_t *)(object + params->subtype_off) = params->subtype_value;
    *(uint64_t *)(object + params->len_off) = (uint64_t)strlen(content);
    assign_string(object + params->wxid_off, wxid);
    assign_string(object + params->content_off, content);

    *(void **)element = object;
    *(void **)(element + 8) = control;
    *(void **)vector = element;
    *(void **)(vector + 8) = element + 16;
    *(void **)(vector + 16) = element + 16;

    uint64_t out[2] __attribute__((aligned(16))) = {0, 0};
    uint64_t controls[3] = {0, 0, 0};
    get_coro(out);
    uint64_t coro = out[0]; controls[0] = out[1];
    if (!coro) return 11;
    out[0] = out[1] = 0;
    get_service(coro, out);
    uint64_t service = out[0]; controls[1] = out[1];
    if (!service) return 21;
    out[0] = out[1] = 0;
    get_context(service, out);
    uint64_t context = out[0]; controls[2] = out[1];
    if (!context) return 31;

    unsigned char scratch[80] __attribute__((aligned(16)));
    memset(scratch, 0xaa, 72);
    memset(scratch + 72, 0, 8);
    do_send(context, scratch, vector, 1);
    if (diagnostics) {
        diagnostics[0] = coro;
        diagnostics[1] = service;
        diagnostics[2] = context;
    }
    release_shared(controls[2]);
    release_shared(controls[1]);
    release_shared(controls[0]);
    return 1;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance; (void)reason; (void)reserved;
    return TRUE;
}
