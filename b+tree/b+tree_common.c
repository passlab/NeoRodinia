#include "b+tree.h"

extern long freeptr;
extern char *mem;
extern long malloc_size;
extern long size;
extern int order;
extern record *krecords;
extern knode *knodes;
extern node *queue;
extern bool verbose_output;

int isInteger(char *str) {

    // Make sure it's not empty
    if (*str == '\0') {
        return 0;
    }

    // If any character is not a digit, return false
    for (; *str != '\0'; str++) {
        if (*str < '0' || *str > '9') {
            return 0;
        }
    }

    // It passed all checks, so it's an integer
    return 1;
}

void list_init(list_t *l, int32_t (*compare)(const void *key, const void *with), void (*datum_delete)(void *)) {
    l->head = l->tail = NULL;
    l->length = 0;
    l->compare = compare;
    l->datum_delete = datum_delete;
}

void list_delete(list_t *l) {
    list_item_t *li, *del;

    for (li = l->head; li;) {
        del = li;
        li = li->next;
        list_item_delete(del, l->datum_delete);
    }

    l->head = l->tail = NULL;
    l->length = 0;
}

void list_reset(list_t *l) {
    list_delete(l);
}

void list_insert_item_head(list_t *l, list_item_t *i) {
    if (l->head) {
        i->next = l->head;
        l->head->pred = i;
        l->head = i;
        l->head->pred = NULL;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;
}

void list_insert_item_tail(list_t *l, list_item_t *i) {
    if (l->head) {
        l->tail->next = i;
        i->pred = l->tail;
        i->next = NULL;
        l->tail = i;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;
}

void list_insert_item_before(list_t *l, list_item_t *next, list_item_t *i) {
    // Assume next is actually in the list! If it's not, we may lose the list.
    if (l->head == next) {
        i->next = next;
        i->pred = NULL;
        l->head = i;
        next->pred = i;
    } else {
        i->next = next;
        i->pred = next->pred;
        next->pred->next = i;
        next->pred = i;
    }
    l->length++;
}

void list_insert_item_after(list_t *l, list_item_t *pred, list_item_t *i) {
    // Assume pred is actually in the list! If it's not, we may lose the list.
    if (l->tail == pred) {
        i->pred = pred;
        i->next = NULL;
        l->tail = i;
        pred->next = i;
    } else {
        i->pred = pred;
        i->next = pred->next;
        pred->next->pred = i;
        pred->next = i;
    }
    l->length++;
}


void list_insert_item_sorted(list_t *l, list_item_t *i) {
    list_item_t *itr;

    if (l->head) {
        for (itr = l->head; itr && l->compare(list_item_get_datum(i), list_item_get_datum(itr)) < 0; itr = itr->next)
            ;
        if (itr) {
            i->next = itr;
            i->pred = itr->pred;
            itr->pred = i;
            i->pred->next = i;
        } else {
            l->tail->next = i;
            i->pred = l->tail;
            i->next = NULL;
            l->tail = i;
        }
    } else {
        l->head = l->tail = i;
        i->pred = i->next = NULL;
    }
    l->length++;
}

void list_insert_head(list_t *l, void *v) {
    list_item_t *i;
    i = (list_item_t *)malloc(sizeof(*i));
    list_item_init(i, v);
    if (l->head) {
        i->next = l->head;
        l->head->pred = i;
        l->head = i;
        l->head->pred = NULL;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;
}

void list_insert_tail(list_t *l, void *v) {
    list_item_t *i;
    i = (list_item_t *)malloc(sizeof(*i));
    list_item_init(i, v);
    if (l->head) {
        l->tail->next = i;
        i->pred = l->tail;
        i->next = NULL;
        l->tail = i;
    } else {
        l->head = l->tail = i;
        i->next = i->pred = NULL;
    }
    l->length++;
}


void list_insert_before(list_t *l, list_item_t *next, void *v) {
    list_item_t *i;

    i = (list_item_t *)malloc(sizeof(*i));
    list_item_init(i, v);

    // Assume next is actually in the list! If it's not, we may lose the list.
    if (l->head == next) {
        i->next = next;
        i->pred = NULL;
        l->head = i;
        next->pred = i;
    } else {
        i->next = next;
        i->pred = next->pred;
        next->pred->next = i;
        next->pred = i;
    }
    l->length++;
}

void list_insert_after(list_t *l, list_item_t *pred, void *v) {
    list_item_t *i;

    i = (list_item_t *)malloc(sizeof(*i));
    list_item_init(i, v);

    // Assume pred is actually in the list! If it's not, we may lose the list.
    if (l->tail == pred) {
        i->pred = pred;
        i->next = NULL;
        l->tail = i;
        pred->next = i;
    } else {
        i->pred = pred;
        i->next = pred->next;
        pred->next->pred = i;
        pred->next = i;
    }
    l->length++;
}

void list_insert_sorted(list_t *l, void *v) {
    list_item_t *itr;
    list_item_t *i;

    i = (list_item_t *)malloc(sizeof(*i));
    list_item_init(i, v);

    if (l->head) {
        for (itr = l->head; itr && l->compare(list_item_get_datum(i), list_item_get_datum(itr)) < 0; itr = itr->next)
            ;
        if (itr) {
            i->next = itr;
            i->pred = itr->pred;
            itr->pred = i;
            i->pred->next = i;
        } else {
            l->tail->next = i;
            i->pred = l->tail;
            i->next = NULL;
            l->tail = i;
        }
    } else {
        l->head = l->tail = i;
        i->pred = i->next = NULL;
    }
    l->length++;
}

void list_remove_item(list_t *l, list_item_t *i) {
    if (i == l->head) {
        l->head = l->head->next;
        if (l->head)
            l->head->pred = NULL;
        else
            l->tail = NULL;
    } else if (i == l->tail) {
        l->tail = l->tail->pred;
        l->tail->next = NULL;
    } else {
        i->pred->next = i->next;
        i->next->pred = i->pred;
    }
    l->length--;
    list_item_delete(i, l->datum_delete);
}

void list_remove_head(list_t *l) {
    list_remove_item(l, l->head);
}

void list_remove_tail(list_t *l) {
    list_remove_item(l, l->tail);
}


list_item_t* list_find_item(list_t *l, void *datum) {
    list_item_t *li;

    for (li = l->head; li && l->compare(datum, list_item_get_datum(li)); li = li->next)
        ;

    return li;
}

list_item_t* list_get_head_item(list_t *l) {
    return l->head;
}

list_item_t* list_get_tail_item(list_t *l) {
    return l->tail;
}

void* list_find(list_t *l, void *datum) {
    list_item_t *li;

    for (li = l->head; li && l->compare(datum, list_item_get_datum(li)); li = li->next)
        ;

    return li ? li->datum : NULL;
}

void* list_get_head(list_t *l) {
    return l->head ? l->head->datum : NULL;
}

void* list_get_tail(list_t *l) {
    return l->tail ? l->tail->datum : NULL;
}

uint32_t list_get_length(list_t *l) {
    return l->length;
}


bool list_is_empty(list_t *l) {
    return (l->length == 0);
}

bool list_not_empty(list_t *l) {
    return (l->length != 0);
}

void list_visit_items(list_t *l, void (*visitor)(void *v)) {
    list_item_t *li;

    for (li = l->head; li; li = li->next)
        visitor(list_item_get_datum(li));
}

void list_item_init(list_item_t *li, void *datum) {
    li->pred = li->next = NULL;
    li->datum = datum;
}

void list_item_delete(list_item_t *li, void (*datum_delete)(void *datum)) {
    if (datum_delete) {
        datum_delete(li->datum);
    }

    free(li);
}

void *list_item_get_datum(list_item_t *li) {
    return li->datum;
}

void list_iterator_init(list_t *l, list_iterator_t *li) {
    *li = l ? l->head : NULL;
}

void list_iterator_delete(list_iterator_t *li) {
    *li = NULL;
}

void list_iterator_next(list_iterator_t *li) {
    if (*li)
        *li = (*li)->next;
}

void list_iterator_prev(list_iterator_t *li) {
    if (*li)
        *li = (*li)->pred;
}

void *list_iterator_get_datum(list_iterator_t *li) {
    return *li ? (*li)->datum : NULL;
}

bool list_iterator_is_valid(list_iterator_t *li) {
    return (*li != NULL);
}

void list_reverse_iterator_init(list_t *l, list_reverse_iterator_t *li) {
    *li = l ? l->tail : NULL;
}

void list_reverse_iterator_delete(list_reverse_iterator_t *li) {
    *li = NULL;
}

void list_reverse_iterator_next(list_reverse_iterator_t *li) {
    if (*li)
        *li = (*li)->pred;
}

void list_reverse_iterator_prev(list_reverse_iterator_t *li) {
    if (*li)
        *li = (*li)->next;
}

void *list_reverse_iterator_get_datum(list_reverse_iterator_t *li) {
    return *li ? (*li)->datum : NULL;
}

bool list_reverse_iterator_is_valid(list_reverse_iterator_t *li) {
    return (li != NULL);
}

//======================================================================================================================================================150
// OUTPUT AND UTILITIES
//======================================================================================================================================================150
void *kmalloc(int size) {
    void *r = (void *)freeptr;
    freeptr += size;
    if (freeptr > malloc_size + (long)mem) {
        printf("Memory Overflow\n");
        exit(1);
    }
    return r;
}

//transforms the current B+ Tree into a single, contiguous block of memory to be used on the GPU
long transform_to_cuda(node *root, bool verbose) {
    struct timeval one, two;
    double time;
    gettimeofday(&one, NULL);
    long max_nodes = (long)(pow(order, log(size) / log(order / 2.0) - 1) + 1);
    malloc_size = size * sizeof(record) + max_nodes * sizeof(knode);
    mem = (char *)malloc(malloc_size);
    if (mem == NULL) {
        printf("Initial malloc error\n");
        exit(1);
    }
    freeptr = (long)mem;

    krecords = (record *)kmalloc(size * sizeof(record));
    knodes = (knode *)kmalloc(max_nodes * sizeof(knode));

    queue = NULL;
    enqueue(root);
    node *n;
    knode *k;
    int i;
    long nodeindex = 0;
    long recordindex = 0;
    long queueindex = 0;
    knodes[0].location = nodeindex++;

    while (queue != NULL) {
        n = dequeue();
        k = &knodes[queueindex];
        k->location = queueindex++;
        k->is_leaf = n->is_leaf;
        k->num_keys = n->num_keys + 2;
        k->keys[0] = INT_MIN;
        k->keys[k->num_keys - 1] = INT_MAX;
        for (i = k->num_keys; i < order; i++) k->keys[i] = INT_MAX;
        if (!k->is_leaf) {
            k->indices[0] = nodeindex++;
            for (i = 1; i < k->num_keys - 1; i++) {
                k->keys[i] = n->keys[i - 1];
                enqueue((node *)n->pointers[i - 1]);
                k->indices[i] = nodeindex++;
            }
            enqueue((node *)n->pointers[i - 1]);
        } else {
            k->indices[0] = 0;
            for (i = 1; i < k->num_keys - 1; i++) {
                k->keys[i] = n->keys[i - 1];
                krecords[recordindex].value = ((record *)n->pointers[i - 1])->value;
                k->indices[i] = recordindex++;
            }
        }

        k->indices[k->num_keys - 1] = queueindex;

        if (verbose) {
            printf("Successfully created knode with index %d\n", k->location);
            printf("Is Leaf: %d, Num Keys: %d\n", k->is_leaf, k->num_keys);
            printf("Pointers: ");
            for (i = 0; i < k->num_keys; i++) printf("%d | ", k->indices[i]);
            printf("\nKeys: ");
            for (i = 0; i < k->num_keys; i++) printf("%d | ", k->keys[i]);
            printf("\n\n");
        }
    }
    long mem_used = size * sizeof(record) + (nodeindex) * sizeof(knode);
    if (verbose) {
        for (i = 0; i < size; i++) printf("%d ", krecords[i].value);
        printf("\nNumber of records = %ld, sizeof(record)=%lu, total=%lu\n", size, sizeof(record), size * sizeof(record));
        printf("Number of knodes = %ld, sizeof(knode)=%lu, total=%lu\n", nodeindex, sizeof(knode), (nodeindex) * sizeof(knode));
        printf("\nDone Transformation. Mem used: %ld\n", mem_used);
    }
    gettimeofday(&two, NULL);
    double oneD = one.tv_sec + (double)one.tv_usec * .000001;
    double twoD = two.tv_sec + (double)two.tv_usec * .000001;
    time = twoD - oneD;
    if (need_full_report()) printf("Tree transformation took %f\n", time);

    return mem_used;
}

list_t *findRange(node *root, int start, int end) {
    int i;
    node *c = find_leaf(root, start, false);

    if (c == NULL) return NULL;

    list_t *retList = (list_t *)malloc(sizeof(list_t));
    list_init(retList, NULL, NULL);

    int counter = 0;
    bool cont = true;
    while (cont && c != 0) {
        cont = false;
        for (i = 0; i < c->num_keys; i++) {
            if (c->keys[i] >= start && c->keys[i] <= end) {
                counter++;
                cont = true;
            } else {
                cont = false;
                break;
            }
        }
        c = (node *)c->pointers[order - 1];
    }
    return retList;
}

/* First message to the user. */
void usage_1(void) {
    printf("B+ Tree of Order %d.\n", order);
    printf("\tAmittai Aviram -- amittai.aviram@yale.edu  Version %s\n", Version);
    printf("\tfollowing Silberschatz, Korth, Sidarshan, Database Concepts, 5th ed.\n\n");
    printf("To build a B+ tree of a different order, start again and enter the order\n");
    printf("as an integer argument:  bpt <order>.  ");
    printf("3 <= order <=20\n");
    printf("To start with input from a file of newline-delimited integers, start again and enter\n");
    printf("the order followed by the filename:  bpt <order> <inputfile>.\n");
}

void usage_2(void) {
    printf("Enter any of the following commands after the prompt > :\n");
    printf("\ti <k>  -- Insert <k> (an integer) as both key and value).\n");
    printf("\tf <k>  -- Find the value under key <k>.\n");
    printf("\tp <k> -- Print the path from the root to key k and its associated value.\n");
    printf("\td <k>  -- Delete key <k> and its associated value.\n");
    printf("\tx -- Destroy the whole tree.  Start again with an empty tree of the same order.\n");
    printf("\tt -- Print the B+ tree.\n");
    printf("\tl -- Print the keys of the leaves (bottom row of the tree).\n");
    printf("\tv -- Toggle output of pointer addresses (\"verbose\") in tree and leaves.\n");
    printf("\tq -- Quit. (Or use Ctl-D.)\n");
    printf("\t? -- Print this help message.\n");
}

/* Helper function for printing the tree out.  See print_tree. */
void enqueue(node* new_node) {
    node *c;
    if (queue == NULL) {
        queue = new_node;
        queue->next = NULL;
    } else {
        c = queue;
        while (c->next != NULL) {
            c = c->next;
        }
        c->next = new_node;
        new_node->next = NULL;
    }
}

node *dequeue(void) {
    node *n = queue;
    queue = queue->next;
    n->next = NULL;
    return n;
}

void print_leaves(node* root) {
    int i;
    node *c = root;
    if (root == NULL) {
        printf("Empty tree.\n");
        return;
    }
    while (!c->is_leaf)
        c = (node *)c->pointers[0];
    while (true) {
        for (i = 0; i < c->num_keys; i++) {
            if (verbose_output)
                printf("%d ", c->keys[i]);
        }
        if (verbose_output)
            if (c->pointers[order - 1] != NULL) {
                printf(" | ");
                c = (node *)c->pointers[order - 1];
            }
        break;
    }
    printf("\n");
}

/* Utility function to give the height of the tree, which length in number of edges of the path from the root to any leaf. */
int height(node* root) {
    int h = 0;
    node *c = root;
    while (!c->is_leaf) {
        c = (node *)c->pointers[0];
        h++;
    }
    return h;
}

int path_to_root(node* root, node* child) {
    int length = 0;
    node *c = child;
    while (c != root) {
        c = c->parent;
        length++;
    }
    return length;
}

/* Prints the B+ tree in the command line in level (rank) order, with the keys in each node and the '|' symbol to separate nodes. With the verbose_output flag set. the values of the pointers corresponding to the keys also appear next to their respective keys, in hexadecimal notation. */
void print_tree(node* root) {
    node *n = NULL;
    int i = 0;
    int rank = 0;
    int new_rank = 0;

    if (root == NULL) {
        printf("Empty tree.\n");
        return;
    }

    queue = NULL;
    enqueue(root);

    while (queue != NULL) {
        n = dequeue();
        if (n->parent != NULL && n == n->parent->pointers[0]) {
            new_rank = path_to_root(root, n);
            if (new_rank != rank) {
                rank = new_rank;
                printf("\n");
            }
        }
        if (verbose_output)
            printf("(%p)", (void *)n);

        for (i = 0; i < n->num_keys; i++) {
            if (verbose_output)
                printf("%p ", (void *)n->pointers[i]);
            printf("%d ", n->keys[i]);
        }

        if (!n->is_leaf) {
            for (i = 0; i <= n->num_keys; i++)
                enqueue((node *)n->pointers[i]);
        }

        if (verbose_output) {
            if (n->is_leaf)
                printf("%p ", (void *)n->pointers[order - 1]);
            else
                printf("%p ", (void *)n->pointers[n->num_keys]);
        }

        printf("| ");
    }
    printf("\n");
}

node *find_leaf(node* root, int key, bool verbose) {
    int i = 0;
    node *c = root;

    if (c == NULL) {
        if (verbose)
            printf("Empty tree.\n");
        return c;
    }

    while (!c->is_leaf) {
        if (verbose) {
            printf("[");
            for (i = 0; i < c->num_keys - 1; i++)
                printf("%d ", c->keys[i]);
            printf("%d] ", c->keys[i]);
        }

        i = 0;
        while (i < c->num_keys) {
            if (key >= c->keys[i])
                i++;
            else
                break;
        }

        if (verbose)
            printf("%d ->\n", i);

        c = (node *)c->pointers[i];
    }

    if (verbose) {
        printf("Leaf [");
        for (i = 0; i < c->num_keys - 1; i++)
            printf("%d ", c->keys[i]);
        printf("%d] ->\n", c->keys[i]);
    }

    return c;
}


/* Finds and returns the record to which a key refers. */
record *find(node* root, int key, bool verbose) {
    int i = 0;
    node *c = find_leaf(root, key, verbose);
    if (c == NULL)
        return NULL;
    for (i = 0; i < c->num_keys; i++)
        if (c->keys[i] == key)
            break;
    if (i == c->num_keys)
        return NULL;
    else
        return (record *)c->pointers[i];
}

/* Finds the appropriate place to split a node that is too big into two. */
int cut(int length) {
    if (length % 2 == 0)
        return length / 2;
    else
        return length / 2 + 1;
}

//======================================================================================================================================================150
// INSERTION
//======================================================================================================================================================150

/* Creates a new record to hold the value to which a key refers. */
record *make_record(int value) {
    record *new_record = (record *)malloc(sizeof(record));
    if (new_record == NULL) {
        perror("Record creation.");
        exit(EXIT_FAILURE);
    } else {
        new_record->value = value;
    }
    return new_record;
}

node *make_node(void) {
    node *new_node;
    new_node = (node *)malloc(sizeof(node));
    if (new_node == NULL) {
        perror("Node creation.");
        exit(EXIT_FAILURE);
    }
    new_node->keys = (int *)malloc((order - 1) * sizeof(int));
    if (new_node->keys == NULL) {
        perror("New node keys array.");
        exit(EXIT_FAILURE);
    }
    new_node->pointers = (void **)malloc(order * sizeof(void *));
    if (new_node->pointers == NULL) {
        perror("New node pointers array.");
        exit(EXIT_FAILURE);
    }
    new_node->is_leaf = false;
    new_node->num_keys = 0;
    new_node->parent = NULL;
    new_node->next = NULL;
    return new_node;
}

/* Creates a new leaf by creating a node and then adapting it appropriately. */
node *make_leaf(void) {
    node *leaf = make_node();
    leaf->is_leaf = true;
    return leaf;
}

int get_left_index(node *parent, node *left) {
    int left_index = 0;
    while (left_index <= parent->num_keys && parent->pointers[left_index] != left)
        left_index++;
    return left_index;
}

node *insert_into_leaf(node *leaf, int key, record *pointer) {
    int i, insertion_point;

    insertion_point = 0;
    while (insertion_point < leaf->num_keys && leaf->keys[insertion_point] < key)
        insertion_point++;

    for (i = leaf->num_keys; i > insertion_point; i--) {
        leaf->keys[i] = leaf->keys[i - 1];
        leaf->pointers[i] = leaf->pointers[i - 1];
    }
    leaf->keys[insertion_point] = key;
    leaf->pointers[insertion_point] = pointer;
    leaf->num_keys++;
    return leaf;
}

/* Inserts a new key and pointer to a new record into a leaf so as to exceed the tree's order, causing the leaf to be split in half. */
node *insert_into_leaf_after_splitting(node *root, node *leaf, int key, record *pointer) {
    node *new_leaf;
    int *temp_keys;
    void **temp_pointers;
    int insertion_index, split, new_key, i, j;

    new_leaf = make_leaf();

    temp_keys = (int *)malloc(order * sizeof(int));
    if (temp_keys == NULL)
    {
        perror("Temporary keys array.");
        exit(EXIT_FAILURE);
    }

    temp_pointers = (void **)malloc(order * sizeof(void *));
    if (temp_pointers == NULL)
    {
        perror("Temporary pointers array.");
        exit(EXIT_FAILURE);
    }

    insertion_index = 0;
    while (insertion_index < order - 1 && leaf->keys[insertion_index] < key)
        insertion_index++;

    for (i = 0, j = 0; i < leaf->num_keys; i++, j++)
    {
        if (j == insertion_index)
            j++;
        temp_keys[j] = leaf->keys[i];
        temp_pointers[j] = leaf->pointers[i];
    }

    temp_keys[insertion_index] = key;
    temp_pointers[insertion_index] = pointer;

    leaf->num_keys = 0;

    split = cut(order - 1);

    for (i = 0; i < split; i++)
    {
        leaf->pointers[i] = temp_pointers[i];
        leaf->keys[i] = temp_keys[i];
        leaf->num_keys++;
    }

    for (i = split, j = 0; i < order; i++, j++)
    {
        new_leaf->pointers[j] = temp_pointers[i];
        new_leaf->keys[j] = temp_keys[i];
        new_leaf->num_keys++;
    }

    free(temp_pointers);
    free(temp_keys);

    new_leaf->pointers[order - 1] = leaf->pointers[order - 1];
    leaf->pointers[order - 1] = new_leaf;

    for (i = leaf->num_keys; i < order - 1; i++)
        leaf->pointers[i] = NULL;
    for (i = new_leaf->num_keys; i < order - 1; i++)
        new_leaf->pointers[i] = NULL;

    new_leaf->parent = leaf->parent;
    new_key = new_leaf->keys[0];

    return insert_into_parent(root, leaf, new_key, new_leaf);
}

node *insert_into_node(node *root, node *n, int left_index, int key, node *right) {
    int i;

    for (i = n->num_keys; i > left_index; i--)
    {
        n->pointers[i + 1] = n->pointers[i];
        n->keys[i] = n->keys[i - 1];
    }
    n->pointers[left_index + 1] = right;
    n->keys[left_index] = key;
    n->num_keys++;
    return root;
}


/* Inserts a new key and pointer to a node into a node, causing the node's size to exceed the order, and causing the node to split into two. */
node *insert_into_node_after_splitting(node *root, node *old_node, int left_index, int key, node *right) {
    int i, j, split, k_prime;
    node *new_node, *child;
    int *temp_keys;
    node **temp_pointers;

    temp_pointers = (node **)malloc((order + 1) * sizeof(node *));
    if (temp_pointers == NULL)
    {
        perror("Temporary pointers array for splitting nodes.");
        exit(EXIT_FAILURE);
    }
    temp_keys = (int *)malloc(order * sizeof(int));
    if (temp_keys == NULL)
    {
        perror("Temporary keys array for splitting nodes.");
        exit(EXIT_FAILURE);
    }

    for (i = 0, j = 0; i < old_node->num_keys + 1; i++, j++)
    {
        if (j == left_index + 1)
            j++;
        temp_pointers[j] = (node *)old_node->pointers[i];
    }

    for (i = 0, j = 0; i < old_node->num_keys; i++, j++)
    {
        if (j == left_index)
            j++;
        temp_keys[j] = old_node->keys[i];
    }

    temp_pointers[left_index + 1] = right;
    temp_keys[left_index] = key;

    split = cut(order);
    new_node = make_node();
    old_node->num_keys = 0;
    for (i = 0; i < split - 1; i++)
    {
        old_node->pointers[i] = temp_pointers[i];
        old_node->keys[i] = temp_keys[i];
        old_node->num_keys++;
    }
    old_node->pointers[i] = temp_pointers[i];
    k_prime = temp_keys[split - 1];
    for (++i, j = 0; i < order; i++, j++)
    {
        new_node->pointers[j] = temp_pointers[i];
        new_node->keys[j] = temp_keys[i];
        new_node->num_keys++;
    }
    new_node->pointers[j] = temp_pointers[i];
    free(temp_pointers);
    free(temp_keys);
    new_node->parent = old_node->parent;
    for (i = 0; i <= new_node->num_keys; i++)
    {
        child = (node *)new_node->pointers[i];
        child->parent = new_node;
    }

    return insert_into_parent(root, old_node, k_prime, new_node);
}


/* Inserts a new node (leaf or internal node) into the B+ tree. Returns the root of the tree after insertion. */
node *insert_into_parent(node *root, node *left, int key, node *right) {
    int left_index;
    node *parent;

    parent = left->parent;

    if (parent == NULL)
        return insert_into_new_root(left, key, right);

    left_index = get_left_index(parent, left);

    if (parent->num_keys < order - 1)
        return insert_into_node(root, parent, left_index, key, right);

    return insert_into_node_after_splitting(root, parent, left_index, key, right);
}


/* Creates a new root for two subtrees and inserts the appropriate key into the new root. */
node *insert_into_new_root(node* left, int key, node* right) {
    node * root = make_node();
    root->keys[0] = key;
    root->pointers[0] = left;
    root->pointers[1] = right;
    root->num_keys++;
    root->parent = NULL;
    left->parent = root;
    right->parent = root;
    return root;
}

node *start_new_tree(int key, record* pointer) {
    node * root = make_leaf();
    root->keys[0] = key;
    root->pointers[0] = pointer;
    root->pointers[order - 1] = NULL;
    root->parent = NULL;
    root->num_keys++;
    return root;
}

/* Master insertion function. Inserts a key and an associated value into the B+ tree, causing the tree to be adjusted however necessary to maintain the B+ tree properties. */
node *insert(node* root, int key, int value) {
    record* pointer;
    node* leaf;

    // The current implementation ignores duplicates.
    if (find(root, key, false) != NULL)
        return root;

    // Create a new record for the value.
    pointer = make_record(value);

    // Case: the tree does not exist yet. Start a new tree.
    if (root == NULL)
        return start_new_tree(key, pointer);

    // Case: the tree already exists. (Rest of function body.)
    leaf = find_leaf(root, key, false);

    // Case: leaf has room for key and pointer.
    if (leaf->num_keys < order - 1) {
        leaf = insert_into_leaf(leaf, key, pointer);
        return root;
    }

    // Case: leaf must be split.
    return insert_into_leaf_after_splitting(root, leaf, key, pointer);
}

//======================================================================================================================================================150
// DELETION
//======================================================================================================================================================150

/* Utility function for deletion. Retrieves the index of a node's nearest neighbor (sibling) to the left if one exists.  If not (the node is the leftmost child), returns -1 to signify this special case. */
int get_neighbor_index(node* n) {
    int i;

    // Return the index of the key to the left of the pointer in the parent pointing to n.
    // If n is the leftmost child, this means return -1.
    for (i = 0; i <= n->parent->num_keys; i++)
        if (n->parent->pointers[i] == n)
            return i - 1;

    // Error state.
    printf("Search for nonexistent pointer to node in parent.\n");
    //printf("Node:  %#x\n", (unsigned int)n);
    exit(EXIT_FAILURE);
}

node* remove_entry_from_node(node* n, int key, node* pointer) {
    int i, num_pointers;

    // Remove the key and shift other keys accordingly.
    i = 0;
    while (n->keys[i] != key)
        i++;
    for (++i; i < n->num_keys; i++)
        n->keys[i - 1] = n->keys[i];

    // Remove the pointer and shift other pointers accordingly.
    // First determine number of pointers.
    num_pointers = n->is_leaf ? n->num_keys : n->num_keys + 1;
    i = 0;
    while (n->pointers[i] != pointer)
        i++;
    for (++i; i < num_pointers; i++)
        n->pointers[i - 1] = n->pointers[i];

    // One key fewer.
    n->num_keys--;

    // Set the other pointers to NULL for tidiness.
    // A leaf uses the last pointer to point to the next leaf.
    if (n->is_leaf)
        for (i = n->num_keys; i < order - 1; i++)
            n->pointers[i] = NULL;
    else
        for (i = n->num_keys + 1; i < order; i++)
            n->pointers[i] = NULL;

    return n;
}

node* adjust_root(node* root) {
    node* new_root;

    // Case: nonempty root.
    // Key and pointer have already been deleted, so nothing to be done.
    if (root->num_keys > 0)
        return root;

    // Case: empty root.
    // If it has a child, promote the first (only) child as the new root.
    if (!root->is_leaf) {
        new_root = (node*)root->pointers[0];
        new_root->parent = NULL;
    } else {
        // If it is a leaf (has no children), then the whole tree is empty.
        new_root = NULL;
    }

    free(root->keys);
    free(root->pointers);
    free(root);

    return new_root;
}


/* Coalesces a node that has become too small after deletion with a neighboring node that can accept the additional entries without exceeding the maximum. */
node* coalesce_nodes(node* root, node* n, node* neighbor, int neighbor_index, int k_prime) {
    int i, j, neighbor_insertion_index, n_start, n_end, new_k_prime;
    node* tmp;
    bool split;
    //Swap neighbor with node if node is on the extreme left and neighbor is to its right.
    if (neighbor_index == -1) {
        tmp = n;
        n = neighbor;
        neighbor = tmp;
    }
    //Starting point in the neighbor for copying keys and pointers from n.
    //Recall that n and neighbor have swapped places in the special case of n being a leftmost child.
    neighbor_insertion_index = neighbor->num_keys;
    split = false;

    if (!n->is_leaf) {
        neighbor->keys[neighbor_insertion_index] = k_prime;
        neighbor->num_keys++;

        n_end = n->num_keys;
        n_start = 0;
        if (n->num_keys + neighbor->num_keys >= order) {
            split = true;
            n_end = cut(order) - 2;
        }

        for (i = neighbor_insertion_index + 1, j = 0; j < n_end; i++, j++) {
            neighbor->keys[i] = n->keys[j];
            neighbor->pointers[i] = n->pointers[j];
            neighbor->num_keys++;
            n->num_keys--;
            n_start++;
        }

        neighbor->pointers[i] = n->pointers[j];

        if (split) {
            new_k_prime = n->keys[n_start];
            for (i = 0, j = n_start + 1; i < n->num_keys; i++, j++) {
                n->keys[i] = n->keys[j];
                n->pointers[i] = n->pointers[j];
            }
            n->pointers[i] = n->pointers[j];
            n->num_keys--;
        }

        for (i = 0; i < neighbor->num_keys + 1; i++) {
            tmp = (node*)neighbor->pointers[i];
            tmp->parent = neighbor;
        }
    } else {
        for (i = neighbor_insertion_index, j = 0; j < n->num_keys; i++, j++) {
            neighbor->keys[i] = n->keys[j];
            neighbor->pointers[i] = n->pointers[j];
            neighbor->num_keys++;
        }
        neighbor->pointers[order - 1] = n->pointers[order - 1];
    }

    if (!split) {
        root = delete_entry(root, n->parent, k_prime, n);
        free(n->keys);
        free(n->pointers);
        free(n);
    } else {
        for (i = 0; i < n->parent->num_keys; i++) {
            if (n->parent->pointers[i + 1] == n) {
                n->parent->keys[i] = new_k_prime;
                break;
            }
        }
    }

    return root;
}


/* Redistributes entries between two nodes when one has become too small after deletion but its neighbor is too big to append the small node's entries without exceeding the maximum */
node* redistribute_nodes(node* root, node* n, node* neighbor, int neighbor_index, int k_prime_index, int k_prime) {
    int i;
    node* tmp;

    if (neighbor_index != -1) {
        if (!n->is_leaf)
            n->pointers[n->num_keys + 1] = n->pointers[n->num_keys];
        for (i = n->num_keys; i > 0; i--) {
            n->keys[i] = n->keys[i - 1];
            n->pointers[i] = n->pointers[i - 1];
        }
        if (!n->is_leaf) {
            n->pointers[0] = neighbor->pointers[neighbor->num_keys];
            tmp = (node*)n->pointers[0];
            tmp->parent = n;
            neighbor->pointers[neighbor->num_keys] = NULL;
            n->keys[0] = k_prime;
            n->parent->keys[k_prime_index] = neighbor->keys[neighbor->num_keys - 1];
        } else {
            n->pointers[0] = neighbor->pointers[neighbor->num_keys - 1];
            neighbor->pointers[neighbor->num_keys - 1] = NULL;
            n->keys[0] = neighbor->keys[neighbor->num_keys - 1];
            n->parent->keys[k_prime_index] = n->keys[0];
        }
    } else {
        if (n->is_leaf) {
            n->keys[n->num_keys] = neighbor->keys[0];
            n->pointers[n->num_keys] = neighbor->pointers[0];
            n->parent->keys[k_prime_index] = neighbor->keys[1];
        } else {
            n->keys[n->num_keys] = k_prime;
            n->pointers[n->num_keys + 1] = neighbor->pointers[0];
            tmp = (node*)n->pointers[n->num_keys + 1];
            tmp->parent = n;
            n->parent->keys[k_prime_index] = neighbor->keys[0];
        }
        for (i = 0; i < neighbor->num_keys; i++) {
            neighbor->keys[i] = neighbor->keys[i + 1];
            neighbor->pointers[i] = neighbor->pointers[i + 1];
        }
        if (!n->is_leaf)
            neighbor->pointers[i] = neighbor->pointers[i + 1];
    }

    n->num_keys++;
    neighbor->num_keys--;

    return root;
}

/* Deletes an entry from the B+ tree. Removes the record and its key and pointer from the leaf, and then makes all appropriate changes to preserve the B+ tree properties. */
node* delete_entry(node* root, node* n, int key, void* pointer) {
    int min_keys;
    node* neighbor;
    int neighbor_index;
    int k_prime_index, k_prime;
    int capacity;

    // Remove key and pointer from node.
    n = remove_entry_from_node(n, key, (node*)pointer);

    // Case: deletion from the root.
    if (n == root)
        return adjust_root(root);

    // Case: deletion from a node below the root.

    // Determine minimum allowable size of node, to be preserved after deletion.
    min_keys = n->is_leaf ? cut(order - 1) : cut(order) - 1;

    // Case: node stays at or above minimum (The simple case).
    if (n->num_keys >= min_keys)
        return root;

    // Case: node falls below minimum. Either coalescence or redistribution is needed.

    // Find the appropriate neighbor node with which to coalesce.
    // Also find the key (k_prime) in the parent between the pointer to node n and the pointer to the neighbor.
    neighbor_index = get_neighbor_index(n);
    k_prime_index = neighbor_index == -1 ? 0 : neighbor_index;
    k_prime = n->parent->keys[k_prime_index];
    neighbor = neighbor_index == -1 ? (node*)n->parent->pointers[1] : (node*)n->parent->pointers[neighbor_index];

    capacity = n->is_leaf ? order : order - 1;

    // Coalescence.
    if (neighbor->num_keys + n->num_keys < capacity)
        return coalesce_nodes(root, n, neighbor, neighbor_index, k_prime);

    // Redistribution.
    else
        return redistribute_nodes(root, n, neighbor, neighbor_index, k_prime_index, k_prime);
}

/* Master deletion function. */
node* deleteVal(node* root, int key) {
    node* key_leaf;
    record* key_record;

    key_record = find(root, key, false);
    key_leaf = find_leaf(root, key, false);
    if (key_record != NULL && key_leaf != NULL) {
        free(key_record);
        root = delete_entry(root, key_leaf, key, key_record);
    }
    return root;
}

void destroy_tree_nodes(node* root) {
    int i;
    if (root->is_leaf) {
        for (i = 0; i < root->num_keys; i++) {
            free(root->pointers[i]);
        }
    } else {
        for (i = 0; i < root->num_keys + 1; i++) {
            destroy_tree_nodes((node*)root->pointers[i]);
        }
    }
    free(root->pointers);
    free(root->keys);
    free(root);
}

node* destroy_tree(node* root) {
    destroy_tree_nodes(root);
    return NULL;
}

