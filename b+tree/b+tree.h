#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <stdbool.h>
#include "utils.h"
#include <limits.h>
#include <stdint.h>

#define OPEN

//=============================================
//    DEFINE
//=============================================

#define fp float
#define Version "1.5"

#ifdef WINDOWS
#define bool char
#define false 0
#define true 1
#endif

#define DEFAULT_ORDER 508
#define NUM_TEAMS 1024
#define TEAM_SIZE 256

#define malloc(size) ({ \
    void *_tmp; \
    if (!(_tmp = malloc(size))) { \
        fprintf(stderr, "Allocation failed at %s:%d!\n", __FILE__, __LINE__); \
        exit(-1); \
    } \
    _tmp; \
})

long long get_time();
int isInteger(char *str);

//=============================================
//    STRUCTURES
//=============================================

typedef struct list_item list_item_t;

typedef struct list_t {
    list_item_t *head, *tail;
    uint32_t length;
    int32_t (*compare)(const void *key, const void *with);
    void (*datum_delete)(void *);
} list_t;

typedef list_item_t *list_iterator_t;
typedef list_item_t *list_reverse_iterator_t;

typedef struct record {
    int value;
} record;

typedef struct node {
    void ** pointers;
    int * keys;
    struct node * parent;
    bool is_leaf;
    int num_keys;
    struct node * next; // Used for queue.
} node;

typedef struct knode {
    int location;
    int indices[DEFAULT_ORDER + 1];
    int keys[DEFAULT_ORDER + 1];
    bool is_leaf;
    int num_keys;
} knode;

struct list_item {
    struct list_item *pred, *next;
    void *datum;
};

//=============================================
//    PROTOTYPES
//=============================================

void list_item_init(list_item_t *li, void *datum);
void list_item_delete(list_item_t *li, void (*datum_delete)(void *datum));
void list_insert_item_tail(list_t *l, list_item_t *i);
void list_insert_item_before(list_t *l, list_item_t *next, list_item_t *i);
void list_insert_item_after(list_t *l, list_item_t *pred, list_item_t *i);
void list_insert_item_sorted(list_t *l, list_item_t *i);

void list_init(list_t *l, int32_t (*compare)(const void *key, const void *with), void (*datum_delete)(void *datum));
void list_delete(list_t *l);
void list_reset(list_t *l);
void list_insert_head(list_t *l, void *v);
void list_insert_tail(list_t *l, void *v);
void list_insert_before(list_t *l, list_item_t *next, void *v);
void list_insert_after(list_t *l, list_item_t *pred, void *v);
void list_insert_sorted(list_t *l, void *v);
void list_insert_item_head(list_t *l, list_item_t *i);
void list_remove_item(list_t *l, list_item_t *i);
void list_remove_head(list_t *l);
void list_remove_tail(list_t *l);
list_item_t *list_find_item(list_t *l, void *datum);
list_item_t *list_get_head_item(list_t *l);
list_item_t *list_get_tail_item(list_t *l);
void *list_find(list_t *l, void *datum);
void *list_get_head(list_t *l);
void *list_get_tail(list_t *l);
uint32_t list_get_length(list_t *l);
bool list_is_empty(list_t *l);
bool list_not_empty(list_t *l);
void list_visit_items(list_t *l, void (*visitor)(void *v));
void *list_item_get_datum(list_item_t *li);
void list_iterator_init(list_t *l, list_iterator_t *li);
void list_iterator_delete(list_iterator_t *li);
void list_iterator_next(list_iterator_t *li);
void list_iterator_prev(list_iterator_t *li);
void *list_iterator_get_datum(list_iterator_t *li);
bool list_iterator_is_valid(list_iterator_t *li);
void list_reverse_iterator_init(list_t *l, list_iterator_t *li);
void list_reverse_iterator_delete(list_iterator_t *li);
void list_reverse_iterator_next(list_iterator_t *li);
void list_reverse_iterator_prev(list_iterator_t *li);
void *list_reverse_iterator_get_datum(list_iterator_t *li);
bool list_reverse_iterator_is_valid(list_reverse_iterator_t *li);

void *kmalloc(int size);
long transform_to_cuda(node *n, bool verbose);
void usage_1(void);
void usage_2(void);
void enqueue(node *new_node);
node *dequeue(void);
int height(node *root);
int path_to_root(node *root, node *child);
void print_leaves(node *root);
void print_tree(node *root);
node *find_leaf(node *root, int key, bool verbose);
record *find(node *root, int key, bool verbose);
int cut(int length);

record *make_record(int value);
node *make_node(void);
node *make_leaf(void);
int get_left_index(node *parent, node *left);
node *insert_into_leaf(node *leaf, int key, record *pointer);
node *insert_into_leaf_after_splitting(node *root, node *leaf, int key, record *pointer);
node *insert_into_node(node *root, node *parent, int left_index, int key, node *right);
node *insert_into_node_after_splitting(node *root, node *parent, int left_index, int key, node *right);
node *insert_into_parent(node *root, node *left, int key, node *right);
node *insert_into_new_root(node *left, int key, node *right);
node *start_new_tree(int key, record *pointer);
node *insert(node *root, int key, int value);

int get_neighbor_index(node *n);
node *adjust_root(node *root);
node *coalesce_nodes(node *root, node *n, node *neighbor, int neighbor_index, int k_prime);
node *redistribute_nodes(node *root, node *n, node *neighbor, int neighbor_index, int k_prime_index, int k_prime);
node *delete_entry(node *root, node *n, int key, void *pointer);
node *deleteVal(node *root, int key);
list_t *findRange(node *root, int start, int end);
void destroy_tree_nodes(node *root);
node *destroy_tree(node *root);

int main(int argc, char *argv[]);

#ifdef __cplusplus
extern "C" {
#endif
extern void kernel_k(record *records, knode *knodes, long knodes_elem, int order, long maxheight, int count, long *currKnode, long *offset, int *keys, record *ans, int threadsPerBlock, long records_elem, long records_mem, long knodes_mem);

extern void kernel_j(knode *knodes, long knodes_elem, long knodes_mem, int order, long maxheight, int count, long *currKnode, long *offset, long *lastKnode, long *offset_2, int *start, int *end, int *recstart, int *reclength, int threadsPerBlock);
#ifdef __cplusplus
}
#endif
