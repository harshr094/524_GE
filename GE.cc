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
    FID_X
};

struct Argument
{
	    Color partition_color;
	    int top_x,top_y;
	    int bottom_x,bottom_y;
	    int size;
	    Argument(int _tx, int _ty, int _bx, int _by, Color _partition, int _size){
	    	top_x = _tx;
	    	top_y = _ty;
	    	bottom_x = _bx;
	    	bottom_y = _by;
	    	partition_color = _partition;
	    	size = _size;
	    }
};

Point<2> make_point(int x, int y) { coord_t vals[2] = { x, y }; return Point<2>(vals); }

int legion_threshold = 4;

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
	int n = 3;
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

    Argument args(0,0,size-1,size-1, partition_color1,size);
    TaskLauncher Populate_launcher(POPULATE_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    Populate_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    Populate_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, Populate_launcher);

    TaskLauncher T_launcher(A_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    T_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    T_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, T_launcher);
}



void a_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    int tx = args.top_x;
    int ty = args.top_y;
    int bx = args.bottom_x;
    int by = args.bottom_y;
    int size = args.size;
    int half_size = size/2;
    if(size <= legion_threshold) {
    	TaskLauncher A_Serial(A_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	A_Serial.add_region_requirement(RegionRequirement(lr,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_Serial.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_Serial);
    }
    else{
    	DomainPointColoring coloring;
    	IndexSpace is = lr.get_index_space();
    	int add = half_size-1;
    	LogicalPartition lp;
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx, ty), make_point(tx+add, ty+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx, ty+half_size), make_point(tx+add, ty+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx+half_size, ty), make_point(tx+half_size+add, ty+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx+half_size, ty+half_size), make_point(tx+half_size+add, ty+half_size+add)));
    		Rect<1>color_space = Rect<1>(0,3);
    		IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    		lp = runtime->get_logical_partition(ctx, lr, ip);
    	}
    	else{
    		 lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
    	}

    	LogicalRegion firstQuad = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    	LogicalRegion secondQuad = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    	LogicalRegion thirdQuad = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    	LogicalRegion fourthQuad = runtime->get_logical_subregion_by_color(ctx, lp, 3);
    
    	Argument Aargs(tx,ty,tx+add,ty+add,args.partition_color,half_size);
    	TaskLauncher A_launcher(A_LEGION_TASK_ID, TaskArgument(&Aargs,sizeof(Argument)));
    	A_launcher.add_region_requirement(RegionRequirement(firstQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_launcher.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher);

    	Argument Bargs(tx,ty+half_size,tx+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher B_laucnher(B_LEGION_TASK_ID,TaskLauncher(&Bargs,sizeof(Argument)));
    	B_laucnher.add_region_requirement(RegionRequirement(secondQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	B_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	B_laucnher.add_field(0,FID_X);
    	B_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,B_laucnher);

    	Argument Cargs(tx+half_size,ty,tx+add+half_size,ty+add,args.partition_color,half_size);
    	TaskLauncher C_laucnher(C_LEGION_TASK_ID,TaskLauncher(&Cargs,sizeof(Argument)));
    	C_laucnher.add_region_requirement(RegionRequirement(thirdQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	C_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	C_laucnher.add_field(0,FID_X);
    	C_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,C_laucnher);

    	Argument Dargs(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher D_launcher(D_LEGION_TASK_ID,TaskLauncher(&Dargs,sizeof(Argument)));
    	D_launcher.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(thirdQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(secondQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_field(0,FID_X);
    	D_launcher.add_field(1,FID_X);
    	D_launcher.add_field(2,FID_X);
    	runtime->execute_task(ctx,D_launcher);

    	Argument Aargs2(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher A_launcher2(A_LEGION_TASK_ID, TaskArgument(&Aargs2,sizeof(Argument)));
    	A_launcher2.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_launcher2.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher2);
	}

}

void b_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalRegion Alr = regions[1].get_logical_region();
    int tx = args.top_x;
    int ty = args.top_y;
    int bx = args.bottom_x;
    int by = args.bottom_y;
    int size = args.size;
    int half_size = size/2;
    if(size <= legion_threshold) {
    	TaskLauncher B_Serial(B_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	B_Serial.add_region_requirement(RegionRequirement(lr,WRITE_DISCARD,EXCLUSIVE,lr));
    	B_Serial.add_region_requirement(RegionRequirement(Alr,READ_ONLY,EXCLUSIVE,Alr));
    	B_Serial.add_field(0,FID_X);
    	B_Serial.add_field(1,FID_X);
    	runtime->execute_task(ctx,B_Serial);
    }
    else{
    	DomainPointColoring coloring;
    	IndexSpace is = lr.get_index_space();
    	int add = half_size-1;
    	LogicalPartition lp;
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx, ty), make_point(tx+add, ty+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx, ty+half_size), make_point(tx+add, ty+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx+half_size, ty), make_point(tx+half_size+add, ty+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx+half_size, ty+half_size), make_point(tx+half_size+add, ty+half_size+add)));
    		Rect<1>color_space = Rect<1>(0,3);
    		IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    		lp = runtime->get_logical_partition(ctx, lr, ip);
    	}
    	else{
    		 lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
    	}

    	LogicalRegion firstQuad = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    	LogicalRegion secondQuad = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    	LogicalRegion thirdQuad = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    	LogicalRegion fourthQuad = runtime->get_logical_subregion_by_color(ctx, lp, 3);

    	is = Alr.get_index_space();
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx, ty-size), make_point(tx+add, ty+add-size)));
    		coloring[1] = Domain(Rect<2>(make_point(tx, ty+half_size-size), make_point(tx+add, ty+half_size+add-size)));
    		coloring[2] = Domain(Rect<2>(make_point(tx+half_size, ty-size), make_point(tx+half_size+add, ty+add-size)));
    		coloring[3] = Domain(Rect<2>(make_point(tx+half_size, ty+half_size-size), make_point(tx+half_size+add, ty+half_size+add-size)));
    		Rect<1>color_space = Rect<1>(0,3);
    		IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    		lp = runtime->get_logical_partition(ctx, lr, ip);
    	}
    	else{
    		 lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
    	}
    	LogicalRegion firstQuadA = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    	LogicalRegion secondQuadA = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    	LogicalRegion thirdQuadA = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    	LogicalRegion fourthQuadA = runtime->get_logical_subregion_by_color(ctx, lp, 3);


    	Argument Firstargs(tx,ty,tx+add,ty+add,args.partition_color,half_size);
    	TaskLauncher First_launcher(B_LEGION_TASK_ID, TaskArgument(&Firstargs,sizeof(Argument)));
    	First_launcher.add_region_requirement(RegionRequirement(firstQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	First_launcher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	First_launcher.add_field(0,FID_X);
    	First_launcher.add_field(1,FID_X);
    	runtime->execute_task(ctx,First_launcher);

    	Argument Secondargs(tx,ty+half_size,tx+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher Second_laucnher(B_LEGION_TASK_ID,TaskLauncher(&Secondargs,sizeof(Argument)));
    	Second_laucnher.add_region_requirement(RegionRequirement(secondQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	Second_laucnher.add_region_requirement(RegionRequirement(secondQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	Second_laucnher.add_field(0,FID_X);
    	Second_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,Second_laucnher);


    	Argument Thirdargs(tx+half_size,ty,tx+add+half_size,ty+add,args.partition_color,half_size);
    	TaskLauncher C_laucnher(C_LEGION_TASK_ID,TaskLauncher(&Cargs,sizeof(Argument)));
    	C_laucnher.add_region_requirement(RegionRequirement(thirdQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	C_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	C_laucnher.add_field(0,FID_X);
    	C_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,C_laucnher);

    	Argument Dargs(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher D_launcher(D_LEGION_TASK_ID,TaskLauncher(&Dargs,sizeof(Argument)));
    	D_launcher.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(thirdQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(secondQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_field(0,FID_X);
    	D_launcher.add_field(1,FID_X);
    	D_launcher.add_field(2,FID_X);
    	runtime->execute_task(ctx,D_launcher);

    	Argument Aargs2(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher A_launcher2(A_LEGION_TASK_ID, TaskArgument(&Aargs2,sizeof(Argument)));
    	A_launcher2.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_launcher2.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher2);
	}

}

void c_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    int tx = args.top_x;
    int ty = args.top_y;
    int bx = args.bottom_x;
    int by = args.bottom_y;
    int size = args.size;
    int half_size = size/2;
    if(size <= legion_threshold) {
    	TaskLauncher A_Serial(A_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	A_Serial.add_region_requirement(RegionRequirement(lr,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_Serial.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_Serial);
    }
    else{
    	DomainPointColoring coloring;
    	IndexSpace is = lr.get_index_space();
    	int add = half_size-1;
    	coloring[0] = Domain(Rect<2>(make_point(tx, ty), make_point(tx+add, ty+add)));
    	coloring[1] = Domain(Rect<2>(make_point(tx, ty+half_size), make_point(tx+add, ty+half_size+add)));
    	coloring[2] = Domain(Rect<2>(make_point(tx+half_size, ty), make_point(tx+half_size+add, ty+add)));
    	coloring[3] = Domain(Rect<2>(make_point(tx+half_size, ty+half_size), make_point(tx+half_size+add, ty+half_size+add)));
    	Rect<1>color_space = Rect<1>(0,3);
    	IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    	LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    	LogicalRegion firstQuad = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    	LogicalRegion secondQuad = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    	LogicalRegion thirdQuad = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    	LogicalRegion fourthQuad = runtime->get_logical_subregion_by_color(ctx, lp, 3);
    
    	Argument Aargs(tx,ty,tx+add,ty+add,args.partition_color,half_size);
    	TaskLauncher A_launcher(A_LEGION_TASK_ID, TaskArgument(&Aargs,sizeof(Argument)));
    	A_launcher.add_region_requirement(RegionRequirement(firstQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_launcher.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher);

    	Argument Bargs(tx,ty+half_size,tx+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher B_laucnher(B_LEGION_TASK_ID,TaskLauncher(&Bargs,sizeof(Argument)));
    	B_laucnher.add_region_requirement(RegionRequirement(secondQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	B_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	B_laucnher.add_field(0,FID_X);
    	B_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,B_laucnher);

    	Argument Cargs(tx+half_size,ty,tx+add+half_size,ty+add,args.partition_color,half_size);
    	TaskLauncher C_laucnher(C_LEGION_TASK_ID,TaskLauncher(&Cargs,sizeof(Argument)));
    	C_laucnher.add_region_requirement(RegionRequirement(thirdQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	C_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	C_laucnher.add_field(0,FID_X);
    	C_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,C_laucnher);

    	Argument Dargs(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher D_launcher(D_LEGION_TASK_ID,TaskLauncher(&Dargs,sizeof(Argument)));
    	D_launcher.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(thirdQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(secondQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_field(0,FID_X);
    	D_launcher.add_field(1,FID_X);
    	D_launcher.add_field(2,FID_X);
    	runtime->execute_task(ctx,D_launcher);

    	Argument Aargs2(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher A_launcher2(A_LEGION_TASK_ID, TaskArgument(&Aargs2,sizeof(Argument)));
    	A_launcher2.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_launcher2.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher2);
	}

}

void d_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    int tx = args.top_x;
    int ty = args.top_y;
    int bx = args.bottom_x;
    int by = args.bottom_y;
    int size = args.size;
    int half_size = size/2;
    if(size <= legion_threshold) {
    	TaskLauncher A_Serial(A_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	A_Serial.add_region_requirement(RegionRequirement(lr,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_Serial.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_Serial);
    }
    else{
    	DomainPointColoring coloring;
    	IndexSpace is = lr.get_index_space();
    	int add = half_size-1;
    	coloring[0] = Domain(Rect<2>(make_point(tx, ty), make_point(tx+add, ty+add)));
    	coloring[1] = Domain(Rect<2>(make_point(tx, ty+half_size), make_point(tx+add, ty+half_size+add)));
    	coloring[2] = Domain(Rect<2>(make_point(tx+half_size, ty), make_point(tx+half_size+add, ty+add)));
    	coloring[3] = Domain(Rect<2>(make_point(tx+half_size, ty+half_size), make_point(tx+half_size+add, ty+half_size+add)));
    	Rect<1>color_space = Rect<1>(0,3);
    	IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    	LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    	LogicalRegion firstQuad = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    	LogicalRegion secondQuad = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    	LogicalRegion thirdQuad = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    	LogicalRegion fourthQuad = runtime->get_logical_subregion_by_color(ctx, lp, 3);
    
    	Argument Aargs(tx,ty,tx+add,ty+add,args.partition_color,half_size);
    	TaskLauncher A_launcher(A_LEGION_TASK_ID, TaskArgument(&Aargs,sizeof(Argument)));
    	A_launcher.add_region_requirement(RegionRequirement(firstQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_launcher.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher);

    	Argument Bargs(tx,ty+half_size,tx+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher B_laucnher(B_LEGION_TASK_ID,TaskLauncher(&Bargs,sizeof(Argument)));
    	B_laucnher.add_region_requirement(RegionRequirement(secondQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	B_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	B_laucnher.add_field(0,FID_X);
    	B_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,B_laucnher);

    	Argument Cargs(tx+half_size,ty,tx+add+half_size,ty+add,args.partition_color,half_size);
    	TaskLauncher C_laucnher(C_LEGION_TASK_ID,TaskLauncher(&Cargs,sizeof(Argument)));
    	C_laucnher.add_region_requirement(RegionRequirement(thirdQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	C_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	C_laucnher.add_field(0,FID_X);
    	C_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,C_laucnher);

    	Argument Dargs(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher D_launcher(D_LEGION_TASK_ID,TaskLauncher(&Dargs,sizeof(Argument)));
    	D_launcher.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(thirdQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(secondQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_field(0,FID_X);
    	D_launcher.add_field(1,FID_X);
    	D_launcher.add_field(2,FID_X);
    	runtime->execute_task(ctx,D_launcher);

    	Argument Aargs2(tx+half_size,ty+half_size,tx+half_size+add,ty+half_size+add,args.partition_color,half_size);
    	TaskLauncher A_launcher2(A_LEGION_TASK_ID, TaskArgument(&Aargs2,sizeof(Argument)));
    	A_launcher2.add_region_requirement(RegionRequirement(fourthQuad,WRITE_DISCARD,EXCLUSIVE,lr));
    	A_launcher2.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher2);
	}

}



void populate_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    const FieldAccessor<WRITE_DISCARD, int, 2> write_acc(regions[0], FID_X);
    for(int i = args.top_x ; i <= args.bottom_x ; i++) {
    	for(int j = args.top_y ; j <= args.bottom_y ; j++ ){
    		write_acc[make_point(i,j)] = rand()%10+1;
    	}
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
        TaskVariantRegistrar registrar(POPULATE_TASK_ID, "populate_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<populate_task>(registrar, "populate_task");
    }
    {
        TaskVariantRegistrar registrar(A_LEGION_TASK_ID, "a_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<a_legion_task>(registrar, "a_legion_task");
    }
    {
        TaskVariantRegistrar registrar(B_LEGION_TASK_ID, "b_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<b_legion_task>(registrar, "b_legion_task");
    }
    {
        TaskVariantRegistrar registrar(C_LEGION_TASK_ID, "c_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<c_legion_task>(registrar, "c_legion_task");
    }
    {
        TaskVariantRegistrar registrar(D_LEGION_TASK_ID, "d_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<d_legion_task>(registrar, "d_legion_task");
    }
    return Runtime::start(argc, argv);
}