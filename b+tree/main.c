#include "b+tree.h"
#include "utils.h"

// Global variables
knode *knodes;
record *krecords;
char *mem;
long freeptr;
long malloc_size;
long size;
long maxheight;
int order = DEFAULT_ORDER; // The order of the B+ tree
bool verbose_output = false; // Toggle verbose output

// Queue for level order printing
node *queue = NULL;

int main(int argc, char** argv ) {
    // Default file names and output file
    char *default_input_file = NULL;
    char *default_command_file = NULL;
    char *output="output.log";
    FILE * pFile;
    double elapsed;
    
    int threadsPerBlock;
    threadsPerBlock = order < 1024 ? order : 1024;
    
    // Input and command file names
    char* input_file = (argc > 1) ? strdup(argv[1]) : strdup(default_input_file);
    char* command_file = (argc > 2) ? strdup(argv[2]) : strdup(default_command_file);
    
    // Print input and command file names if verbose
    if (need_full_report()) {
        printf("Input File: %s \n", input_file);
        printf("Command File: %s \n", command_file);
    }
    
    // Read command file into buffer
    FILE * commandFile;
    long lSize;
    char * commandBuffer;
    size_t result;
    
    commandFile = fopen ( command_file, "rb" );
    if (commandFile==NULL) {fputs ("Command File error",stderr); exit (1);}
    
    fseek (commandFile , 0 , SEEK_END);
    lSize = ftell (commandFile);
    rewind (commandFile);
    
    commandBuffer = (char*) malloc (sizeof(char)*lSize);
    if (commandBuffer == NULL) {fputs ("Command Buffer memory error",stderr); exit (2);}
    
    result = fread (commandBuffer,1,lSize,commandFile);
    if (result != lSize) {fputs ("Command file reading error",stderr); exit (3);}
    
    fclose (commandFile);
    
    // Print command buffer if verbose
    if (need_full_report()) {
        printf("Command Buffer: \n");
        printf("%s\n",commandBuffer);
    }
    // Open output file for writing
    pFile = fopen(output, "w+");
    if (pFile == NULL) {
        fprintf(stderr, "Failed to open %s!\n", output);
        exit(1);
    }
    
    // Write a starting message to the output file
    fprintf(pFile, "******starting******\n");
    fclose(pFile);
    
    // General variables
    FILE *file_pointer;
    node *root;
    root = NULL;
    record *r;
    int input;
    char instruction;
    order = DEFAULT_ORDER;
    verbose_output = false;
    
    // Get input from file, if file provided
    if (input_file != NULL) {
        if (need_full_report()) {
            printf("Getting input from file %s...\n", argv[1]);
        }
        
        // Open input file
        file_pointer = fopen(input_file, "r");
        if (file_pointer == NULL) {
            perror("Failure to open input file.");
            exit(EXIT_FAILURE);
        }
        
        // Get the number of numbers in the file
        fscanf(file_pointer, "%d\n", &input);
        size = input;
        
        // Save all numbers into the B+ tree
        while (!feof(file_pointer)) {
            fscanf(file_pointer, "%d\n", &input);
            root = insert(root, input, input);
        }
        
        // Close the file
        fclose(file_pointer);
    } else {
        // If no input file provided, return 0
        return 0;
    }
    
    // Get tree statistics
    if (need_full_report()) {
        printf("Transforming data to a GPU suitable structure...\n");
    }
    long mem_used = transform_to_cuda(root, 0);
    maxheight = height(root);
    long rootLoc = (long) knodes - (long) mem;
    
    // Process commands
    char *commandPointer = commandBuffer;
    if (need_full_report()) {
        printf("Waiting for command\n");
        printf("> ");
    }
    while (sscanf(commandPointer, "%c", &instruction) != EOF) {
        commandPointer++;
        switch (instruction) {
            // ----------------------------------------40
            // Insert
            // ----------------------------------------40
            case 'i':
            {
                scanf("%d", &input);
                while (getchar() != (int)'\n');
                root = insert(root, input, input);
                print_tree(root);
                break;
            }
                
            // ----------------------------------------40
            // find
            // ----------------------------------------40
            case 'p':
            {
                scanf("%d", &input);
                while (getchar() != (int)'\n');
                r = find(root, input, instruction == 'p');
                if (r == NULL)
                    printf("Record not found under key %d.\n", input);
                else
                    printf("Record found: %d\n",r->value);
                break;
            }
                
            // ----------------------------------------40
            // delete value
            // ----------------------------------------40
            case 'd':
            {
                scanf("%d", &input);
                while (getchar() != (int)'\n');
                root = (node *) deleteVal(root, input);
                print_tree(root);
                break;
            }
                
            // ----------------------------------------40
            // destroy tree
            // ----------------------------------------40
            case 'x':
            {
                while (getchar() != (int)'\n');
                root = destroy_tree(root);
                print_tree(root);
                break;
            }
            // ----------------------------------------40
            // print leaves
            // ----------------------------------------40
            case 'l':
            {
                while (getchar() != (int)'\n');
                print_leaves(root);
                break;
            }
                
            // ----------------------------------------40
            // print tree
            // ----------------------------------------40
            case 't':
            {
                while (getchar() != (int)'\n');
                print_tree(root);
                break;
            }
                
            // ----------------------------------------40
            // toggle verbose output
            // ----------------------------------------40
            case 'v':
            {
                while (getchar() != (int)'\n');
                verbose_output = !verbose_output;
                break;
            }
                
            // ----------------------------------------40
            // quit
            // ----------------------------------------40
            case 'q':
            {
                while (getchar() != (int)'\n');
                return EXIT_SUCCESS;
            }
            // ----------------------------------------40
            // find K (initK, findK)
            // ----------------------------------------40
            case 'k':
            {
                // Get the number of queries from the user
                int count;
                sscanf(commandPointer, "%d", &count);
                while (*commandPointer != ' ' && *commandPointer != '\n')
                    commandPointer++;
                
                if (need_full_report())
                    printf("\n ******command: k count=%d \n", count);
                
                if (count > 65535) {
                    printf("ERROR: Number of requested queries should be 65,535 at most. (limited by # of CUDA blocks)\n");
                    exit(0);
                }
                
                record *records = (record *)mem;
                long records_elem = (long)rootLoc / sizeof(record);
                long records_mem = (long)rootLoc;
                // INPUT: knodes CPU allocation (setting pointer in mem variable)
                knode *knodes = (knode *)((long)mem + (long)rootLoc);
                long knodes_elem = ((long)(mem_used) - (long)rootLoc) / sizeof(knode);
                long knodes_mem = (long)(mem_used) - (long)rootLoc;
                
                // INPUT: currKnode CPU allocation
                long *currKnode;
                currKnode = (long *)malloc(count * sizeof(long));
                // INPUT: offset CPU initialization
                memset(currKnode, 0, count * sizeof(long));
                
                // INPUT: offset CPU allocation
                long *offset;
                offset = (long *)malloc(count * sizeof(long));
                // INPUT: offset CPU initialization
                memset(offset, 0, count * sizeof(long));
                
                // INPUT: keys CPU allocation
                int *keys;
                keys = (int *)malloc(count * sizeof(int));
                // INPUT: keys CPU initialization
                int i;
                for (i = 0; i < count; i++) {
                    keys[i] = (rand() / (float)RAND_MAX) * size;
                }
                
                // OUTPUT: ans CPU allocation
                record *ans = (record *)malloc(sizeof(record) * count);
                // OUTPUT: ans CPU initialization
                for (i = 0; i < count; i++) {
                    ans[i].value = -1;
                }
                
                elapsed = read_timer();
                
                kernel_k(records, knodes, knodes_elem, order, maxheight, count, currKnode, offset, keys, ans, threadsPerBlock, records_elem, records_mem, knodes_mem);
                
                elapsed = (read_timer() - elapsed);
                pFile = fopen(output, "aw+");
                if (pFile == NULL) {
                    fprintf(stderr, "Fail to open %s !\n", output);
                }
                
                fprintf(pFile, "\n ******command: k count=%d \n", count);
                for (i = 0; i < count; i++) {
                    fprintf(pFile, "%d    %d\n", i, ans[i].value);
                }
                fprintf(pFile, " \n");
                fclose(pFile);
                
                // Free memory
                free(currKnode);
                free(offset);
                free(keys);
                free(ans);
                
                // Break out of case
                break;
            }
                
            // ----------------------------------------40
            // find range
            // ----------------------------------------40
            case 'r':
            {
                int start, end;
                scanf("%d", &start);
                scanf("%d", &end);
                if (start > end) {
                    int temp = start;
                    start = end;
                    end = temp;
                }
                printf("For range %d to %d, ", start, end);
                list_t *ansList;
                ansList = findRange(root, start, end);
                printf("%d records found\n", list_get_length(ansList));
                free(ansList);
                break;
                
            }
            // ----------------------------------------40
            // find Range K (initK, findRangeK)
            // ----------------------------------------40
            case 'j':
            {
                // Get the number of queries and range size from the user
                int count, rSize;
                sscanf(commandPointer, "%d", &count);
                while (*commandPointer != ' ' && *commandPointer != '\n')
                    commandPointer++;
                
                sscanf(commandPointer, "%d", &rSize);
                while (*commandPointer != ' ' && *commandPointer != '\n')
                    commandPointer++;
                
                if (need_full_report())
                    printf("\n******command: j count=%d, rSize=%d \n", count, rSize);
                
                if (rSize > size || rSize < 0) {
                    printf("Search range size is larger than data set size %d.\n", (int)size);
                    exit(0);
                }
                
                // INPUT: knodes CPU allocation (setting pointer in mem variable)
                knode *knodes = (knode *)((long)mem + (long)rootLoc);
                long knodes_elem = ((long)(mem_used) - (long)rootLoc) / sizeof(knode);
                long knodes_mem = (long)(mem_used) - (long)rootLoc;
                // INPUT: currKnode CPU allocation
                long *currKnode;
                currKnode = (long *)malloc(count * sizeof(long));
                // INPUT: offset CPU initialization
                memset(currKnode, 0, count * sizeof(long));
                
                // INPUT: offset CPU allocation
                long *offset;
                offset = (long *)malloc(count * sizeof(long));
                // INPUT: offset CPU initialization
                memset(offset, 0, count * sizeof(long));
                
                // INPUT: lastKnode CPU allocation
                long *lastKnode;
                lastKnode = (long *)malloc(count * sizeof(long));
                // INPUT: offset CPU initialization
                memset(lastKnode, 0, count * sizeof(long));
                
                // INPUT: offset_2 CPU allocation
                long *offset_2;
                offset_2 = (long *)malloc(count * sizeof(long));
                // INPUT: offset CPU initialization
                memset(offset_2, 0, count * sizeof(long));
                
                // INPUT: start, end CPU allocation
                int *start;
                start = (int *)malloc(count * sizeof(int));
                int *end;
                end = (int *)malloc(count * sizeof(int));
                // INPUT: start, end CPU initialization
                int i;
                for (i = 0; i < count; i++) {
                    start[i] = (rand() / (float)RAND_MAX) * size;
                    end[i] = start[i] + rSize;
                    if (end[i] >= size) {
                        start[i] = start[i] - (end[i] - size);
                        end[i] = size - 1;
                    }
                }
                
                // INPUT: recstart, reclength CPU allocation
                int *recstart;
                recstart = (int *)malloc(count * sizeof(int));
                int *reclength;
                reclength = (int *)malloc(count * sizeof(int));
                // OUTPUT: ans CPU initialization
                for (i = 0; i < count; i++) {
                    recstart[i] = 0;
                    reclength[i] = 0;
                }
                
                elapsed = read_timer();
                
                kernel_j(knodes, knodes_elem, knodes_mem, order, maxheight, count,
                         currKnode, offset, lastKnode, offset_2,
                         start, end, recstart, reclength, threadsPerBlock);
                
                elapsed = (read_timer() - elapsed);
                
                pFile = fopen(output, "aw+");
                if (pFile == NULL) {
                    fprintf(stderr, "Fail to open %s !\n", output);
                }
                
                fprintf(pFile, "\n******command: j count=%d, rSize=%d \n", count, rSize);
                for (i = 0; i < count; i++) {
                    fprintf(pFile, "%d    %d    %d\n", i, recstart[i], reclength[i]);
                }
                fprintf(pFile, " \n");
                fclose(pFile);
                
                // Free memory
                free(currKnode);
                free(offset);
                free(lastKnode);
                free(offset_2);
                free(start);
                free(end);
                free(recstart);
                free(reclength);
                
                // Break out of case
                break;
            }
        }
    }
    if (need_full_report()) {
        printf("> \n");
        printf("Total time: %4f\n", elapsed * 1.0e3);
    } else {
        printf("%4f\n", elapsed * 1.0e3);
    }
}

