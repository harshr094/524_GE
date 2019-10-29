#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <cstdio>
#include "legion.h"
#include <vector>
#include <queue>
#include <utility>

using namespace Legion;
using namespace std;

enum TASK_IDs
{
	TOP_LEVEL_TASK_ID,
	GE_TASK_ID,
	A_LEGION_TASK_ID,
	B_LEGION_TASK_ID,
	C_LEGION_TASK_ID,
	D_LEGION_TASK_ID,
	A_NON_LEGION_TASK_ID,
	B_NON_LEGION_TASK_ID,
	C_NON_LEGION_TASK_ID,
	D_NON_LEGION_TASK_ID,
};

enum FieldId{
    FID_X,
    FID_Y
};

struct Argument
{
	    Color partition_color;
	    int top_x,top_y;
	    int bottom_x,bottom_y;
	    int rows,cols;
	    Argument(int _tx, int _ty, int _bx, int _by, int _r, int _c, Color _partition){
	    	top_x = _tx;
	    	top_y = _ty;
	    	bottom_x = _bx;
	    	bottom_y = _by;
	    	rows = _r;
	    	cols = _c;
	    	partition_color = _partition;
	    }
};

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
	int n = 10;
	int size = (1<<n);
	Rect<1> rect(0LL, size);
	IndexSpace is = runtime->create_index_space(ctx, rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(int), FID_X);
        allocator.allocate_field(sizeof(int), FID_Y);
    }
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Color partition_color1 = 10;
    Argument args(0,0,size-1,size-1, n, n, partition_color1);
    TaskLauncher T_launcher(GE_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    T_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    T_launcher.add_field(0, FID_X);
    T_launcher.add_field(1,FID_Y);
    runtime->execute_task(ctx, T_launcher);
}

void ge_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){

}

int main(int argc,char** argv){
	srand(time(NULL));
	Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
	{
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    {
        TaskVariantRegistrar registrar(GE_TASK_ID, "ge_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<ge_task>(registrar, "ge_level");
    }

    return Runtime::start(argc, argv);
}