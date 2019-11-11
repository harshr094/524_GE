#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <cstdio>
#include "legion.h"
#include <vector>
#include <queue>
#include <utility>
#include "legion_domain.h"

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
	POPULATE_TASK_ID,
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
	    Argument(int _tx, int _ty, int _bx, int _by, Color _partition){
	    	top_x = _tx;
	    	top_y = _ty;
	    	bottom_x = _bx;
	    	bottom_y = _by;
	    	partition_color = _partition;
	    }
};

Point<2> make_point(int x, int y) { coord_t vals[2] = { x, y }; return Point<2>(vals); }

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
	int n = 10;
	int size = (1<<n);
	Domain domain = Domain(Rect<2>(make_point(0, 0), make_point(size - 1, size - 1)));
  	IndexSpace is = runtime->create_index_space(ctx, domain);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(int), FID_X);
    }
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Color partition_color1 = 10;

    Argument args(0,0,size-1,size-1, partition_color1);
    TaskLauncher T_launcher(GE_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    T_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    T_launcher.add_field(0, FID_X);
    T_launcher.add_field(1,FID_Y);
    runtime->execute_task(ctx, T_launcher);

    Argument Printargs(0,0,size-1,size-1, partition_color1);
    TaskLauncher Print_launcher(POPULATE_TASK_ID, TaskArgument(&Printargs,sizeof(Argument)));
    Print_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    Print_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, Print_launcher);
}

void ge_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    int tx = args.top_x;
    int ty = args.top_y;
    int bx = args.bottom_x;
    int by = args.bottom_y;
 	int midx = tx+bx;
 	midx/=2;
 	int midy = ty + by;
 	midy/=2;
    DomainPointColoring coloring;
    IndexSpace is = lr.get_index_space();

    coloring[0] = Domain(Rect<2>(make_point(tx, ty), make_point(midx-1, midy-1)));
    coloring[1] = Domain(Rect<2>(make_point(tx, midy), make_point(midx-1, by)));
    coloring[2] = Domain(Rect<2>(make_point(midx, ty), make_point(bx, midy-1)));
    coloring[3] = Domain(Rect<2>(make_point(midx, midy), make_point(bx, by)));
    Rect<1>color_space = Rect<1>(0,3);
    IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    
    LogicalRegion firstQuad = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    LogicalRegion secondQuad = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    LogicalRegion thirdQuad = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    LogicalRegion fourthQuad = runtime->get_logical_subregion_by_color(ctx, lp, 3);

    Argument Aargs(tx,ty,midx-1,midy-1,args.partition_color);
    TaskLauncher A_launcher(A_LEGION_TASK_ID, TaskArgument(&Aargs,sizeof(Argument)));
    A_launcher.add_region_requirement(RegionRequirement(firstQuad,WRITE_DISCARD,EXCLUSIVE,firstQuad));
    A_launcher.add_field(0,FID_X);
    A_launcher.add_field(1,FID_Y);
    runtime->execute_task(ctx,A_launcher);
}


void a_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
}

void populate_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    const FieldAccessor<WRITE_DISCARD, int, 1> write_acc(regions[0], FID_X);
    for(int i = args.top_x ; i <= args.bottom_x ; i++) {
    	for(int j = args.top_y ; j <= args.bottom_y ; j++ ){
    		write_acc[make_point(i,j)] = rand()%10;
    	}
    }
    const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
    for(int i = args.top_x ; i <= args.bottom_x ; i++) {
    	for(int j = args.top_y ; j <= args.bottom_y ; j++ ){
    		 cout<<i<<"~"<<j<<"~"<<read_acc[make_point(i,j)]<<" ";
    	}cout<<endl;
    }
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
    {
        TaskVariantRegistrar registrar(POPULATE_TASK_ID, "populate_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<populate_task>(registrar, "populate_task");
    }
    {
        TaskVariantRegistrar registrar(A_LEGION_TASK_ID, "a_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<a_legion_task>(registrar, "a_legion_task");
    }
    return Runtime::start(argc, argv);
}