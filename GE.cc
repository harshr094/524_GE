#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <cstdio>
#include "legion.h"
#include <vector>
#include <queue>
#include <utility>
#include <string>
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
	PRINT_TASK_ID
};

enum FieldId{
    FID_X
};

struct SingleMat
{
	    Color partition_color;
	    int top_x,top_y;
	    int bottom_x,bottom_y;
	    int size;
	    SingleMat(int _tx, int _ty, int _bx, int _by, Color _partition, int _size){
	    	top_x = _tx;
	    	top_y = _ty;
	    	bottom_x = _bx;
	    	bottom_y = _by;
	    	partition_color = _partition;
	    	size = _size;
	    }
};


struct Argument
{
	    Color partition_color;
	    int top_x1, top_y1;  // X
	   	int top_x2, top_y2;  // U
	    int top_x3, top_y3;  // V
	    int top_x4, top_y4;  // W
	    int size;
	    Argument(int _tx1, int _ty1, int _tx2, int _ty2, int _tx3, int _ty3, int _tx4, int _ty4, Color _partition, int _size){
		top_x1 = _tx1; top_y1 = _ty1; 
		top_x2 = _tx2; top_y2 = _ty2; 
		top_x3 = _tx3; top_y3 = _ty3; 
		top_x4 = _tx4; top_y4 = _ty4;
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
        allocator.allocate_field(sizeof(double), FID_X);
    }
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Color partition_color1 = 10;
    SingleMat args(0,0,size-1,size-1, partition_color1,size);
    TaskLauncher Populate_launcher(POPULATE_TASK_ID, TaskArgument(&args,sizeof(SingleMat)));
    Populate_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    Populate_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, Populate_launcher);
    Argument GEarg(0,0,0,0,0,0,0,0,partition_color1,size);
    TaskLauncher T_launcher(A_LEGION_TASK_ID, TaskArgument(&GEarg,sizeof(Argument)));
    T_launcher.add_region_requirement(RegionRequirement(lr1, READ_WRITE, EXCLUSIVE, lr1));
    T_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, T_launcher);
    TaskLauncher Print_launcher(PRINT_TASK_ID, TaskArgument(&args,sizeof(SingleMat)));
    Print_launcher.add_region_requirement(RegionRequirement(lr1,READ_ONLY,EXCLUSIVE,lr1));
    Print_launcher.add_field(0,FID_X);
    runtime->execute_task(ctx,Print_launcher);
}


void a_non_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    int size = args.size;
    const FieldAccessor<READ_WRITE, double, 2> write_acc(regions[0], FID_X);
    for(int k = args.top_x4; k < args.top_x4 + size; k++){
    	for(int i = args.top_x1; i < args.top_x1 + size; i++){
    		for(int j = args.top_y1; j < args.top_y1 + size; j++){
    			if((k<i)&&(k<j)){
    				write_acc[make_point(i,j)] = write_acc[make_point(i,j)] - (write_acc[make_point(i,k)]/write_acc[make_point(k,k)])*write_acc[make_point(k,j)];
    			}
    		}
    	}
    }
}

void b_non_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    int size = args.size;
    const FieldAccessor<READ_WRITE, double, 2> write_acc(regions[0], FID_X);
    const FieldAccessor<READ_ONLY, double, 2> read_acc(regions[1], FID_X);
    for(int k = args.top_x4; k < args.top_x4 + size; k++){
    	for(int i = args.top_x1; i < args.top_x1 + size; i++){
    		for(int j = args.top_y1; j < args.top_y1 + size; j++){
    			if((k<i)&&(k<j)){
    				write_acc[make_point(i,j)] = write_acc[make_point(i,j)] - (read_acc[make_point(i,k)]/write_acc[make_point(k,k)])*write_acc[make_point(k,j)];
    			}
    		}
    	}
    }
}

void c_non_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    int size = args.size;
    const FieldAccessor<READ_WRITE, double, 2> write_acc(regions[0], FID_X);
    const FieldAccessor<READ_ONLY, double, 2> read_acc(regions[1], FID_X);
    for(int k = args.top_x4; k < args.top_x4 + size; k++){
    	for(int i = args.top_x1; i < args.top_x1 + size; i++){
    		for(int j = args.top_y1; j < args.top_y1 + size; j++){
    			if((k<i)&&(k<j)){
    				write_acc[make_point(i,j)] = write_acc[make_point(i,j)] - (write_acc[make_point(i,k)]/write_acc[make_point(k,k)])*read_acc[make_point(k,j)];
    			}
    		}
    	}
    }
}


void d_non_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    int size = args.size;
    const FieldAccessor<READ_WRITE, double, 2> write_acc(regions[0], FID_X);
    const FieldAccessor<READ_ONLY, double, 2> read_acc(regions[1], FID_X);
    const FieldAccessor<READ_ONLY, double, 2> read_acc2(regions[2], FID_X);
    const FieldAccessor<READ_ONLY, double, 2> read_acc3(regions[3], FID_X);
    for(int k = args.top_x4; k < args.top_x4 + size; k++){
    	for(int i = args.top_x1; i < args.top_x1 + size; i++){
    		for(int j = args.top_y1; j < args.top_y1 + size; j++){
    			if((k<i)&&(k<j)){
    				write_acc[make_point(i,j)] = write_acc[make_point(i,j)] - (read_acc[make_point(i,k)]/read_acc3[make_point(k,k)])*read_acc2[make_point(k,j)];
    			}
    		}
    	}
    }
}


void a_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    int size = args.size;
    int half_size = size/2;
    if(size <= legion_threshold) {
    	TaskLauncher A_Serial(A_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	A_Serial.add_region_requirement(RegionRequirement(lr,READ_WRITE,EXCLUSIVE,lr));
    	A_Serial.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_Serial);
    }
    else{
    	int tx = args.top_x1;
    	int ty = args.top_y1;
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
    
    	Argument Aargs1(args.top_x1,args.top_y1,args.top_x2, args.top_y2, args.top_x3, args.top_y3,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher A_launcher(A_LEGION_TASK_ID, TaskArgument(&Aargs1,sizeof(Argument)));
    	A_launcher.add_region_requirement(RegionRequirement(firstQuad,READ_WRITE,EXCLUSIVE,lr));
    	A_launcher.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher);

    	Argument Bargs(tx,ty+half_size,args.top_x2,args.top_y2,args.top_x3,args.top_y3+half_size,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher B_laucnher(B_LEGION_TASK_ID,TaskArgument(&Bargs,sizeof(Argument)));
    	B_laucnher.add_region_requirement(RegionRequirement(secondQuad,READ_WRITE,EXCLUSIVE,lr));
    	B_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	B_laucnher.add_field(0,FID_X);
    	B_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,B_laucnher);

    	Argument Cargs(tx+half_size,ty,args.top_x2+half_size,args.top_y2,args.top_x3,args.top_y3,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher C_laucnher(C_LEGION_TASK_ID,TaskArgument(&Cargs,sizeof(Argument)));
    	C_laucnher.add_region_requirement(RegionRequirement(thirdQuad,READ_WRITE,EXCLUSIVE,lr));
    	C_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	C_laucnher.add_field(0,FID_X);
    	C_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,C_laucnher);

    	Argument Dargs(tx+half_size,ty+half_size,args.top_x2+half_size,args.top_y2,args.top_x3,args.top_y3+half_size,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher D_launcher(D_LEGION_TASK_ID,TaskArgument(&Dargs,sizeof(Argument)));
    	D_launcher.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(thirdQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(secondQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_launcher.add_field(0,FID_X);
    	D_launcher.add_field(1,FID_X);
    	D_launcher.add_field(2,FID_X);
    	D_launcher.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_launcher);

    	Argument Aargs2(tx+half_size,ty+half_size,args.top_x2+half_size,args.top_y2+half_size,args.top_x3+half_size,args.top_y3+half_size,args.top_x4+half_size,args.top_y4+half_size,args.partition_color,half_size);
    	TaskLauncher A_launcher2(A_LEGION_TASK_ID, TaskArgument(&Aargs2,sizeof(Argument)));
    	A_launcher2.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	A_launcher2.add_field(0,FID_X);
    	runtime->execute_task(ctx,A_launcher2);
	}

}

void b_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalRegion Alr = regions[1].get_logical_region();    
    int size = args.size;
    int half_size = size/2;
    if(size <= legion_threshold) {
    	TaskLauncher B_Serial(B_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	B_Serial.add_region_requirement(RegionRequirement(lr,READ_WRITE,EXCLUSIVE,lr));
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
    	int tx1 = args.top_x1;
    	int ty1 = args.top_y1;
    	int tx2 = args.top_x2;
    	int ty2 = args.top_y2;
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx1, ty1), make_point(tx1+add, ty1+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx1, ty1+half_size), make_point(tx1+add, ty1+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx1+half_size, ty1), make_point(tx1+half_size+add, ty1+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx1+half_size, ty1+half_size), make_point(tx1+half_size+add, ty1+half_size+add)));
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
    		coloring[0] = Domain(Rect<2>(make_point(tx2, ty2), make_point(tx2+add, ty2+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx2, ty2+half_size), make_point(tx2+add, ty2+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx2+half_size, ty2), make_point(tx2+half_size+add, ty2+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx2+half_size, ty2+half_size), make_point(tx2+half_size+add, ty2+half_size+add)));
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


    	Argument Firstargs(tx1,ty1,tx2,ty2,args.top_x3,args.top_y3,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher First_launcher(B_LEGION_TASK_ID, TaskArgument(&Firstargs,sizeof(Argument)));
    	First_launcher.add_region_requirement(RegionRequirement(firstQuad,READ_WRITE,EXCLUSIVE,lr));
    	First_launcher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	First_launcher.add_field(0,FID_X);
    	First_launcher.add_field(1,FID_X);
    	runtime->execute_task(ctx,First_launcher);

    	Argument Secondargs(tx1,ty1+half_size,tx2,ty2,args.top_x3,args.top_y3+half_size,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher Second_laucnher(B_LEGION_TASK_ID,TaskArgument(&Secondargs,sizeof(Argument)));
    	Second_laucnher.add_region_requirement(RegionRequirement(secondQuad,READ_WRITE,EXCLUSIVE,lr));
    	Second_laucnher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	Second_laucnher.add_field(0,FID_X);
    	Second_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,Second_laucnher);


    	Argument Dargs1(tx1+half_size,ty1,tx2+half_size,ty2,args.top_x3,args.top_y3,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher D_laucnher(D_LEGION_TASK_ID,TaskArgument(&Dargs1,sizeof(Argument)));
    	D_laucnher.add_region_requirement(RegionRequirement(thirdQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher.add_region_requirement(RegionRequirement(thirdQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher.add_field(0,FID_X);
    	D_laucnher.add_field(1,FID_X);
    	D_laucnher.add_field(2,FID_X);
    	D_laucnher.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher);

    	Argument Dargs2(tx1+half_size,ty1+half_size,tx2+half_size,ty2,args.top_x3,args.top_y3+half_size,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher D_laucnher2(D_LEGION_TASK_ID,TaskArgument(&Dargs2,sizeof(Argument)));
    	D_laucnher2.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher2.add_region_requirement(RegionRequirement(thirdQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher2.add_region_requirement(RegionRequirement(secondQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_laucnher2.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher2.add_field(0,FID_X);
    	D_laucnher2.add_field(1,FID_X);
    	D_laucnher2.add_field(2,FID_X);
    	D_laucnher2.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher2);

    	Argument Thirdargs(tx1+half_size,ty1,tx2+half_size,ty2+half_size,args.top_x3+half_size,args.top_y3,args.top_x4+half_size,args.top_y4+half_size,args.partition_color,half_size);
    	TaskLauncher Third_launcher(B_LEGION_TASK_ID, TaskArgument(&Thirdargs,sizeof(Argument)));
    	Third_launcher.add_region_requirement(RegionRequirement(thirdQuad,READ_WRITE,EXCLUSIVE,lr));
    	Third_launcher.add_region_requirement(RegionRequirement(fourthQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	Third_launcher.add_field(0,FID_X);
    	Third_launcher.add_field(1,FID_X);
    	runtime->execute_task(ctx,Third_launcher);

    	Argument Fourthargs(tx1+half_size,ty1+half_size,tx2+half_size,ty2+half_size,args.top_x3+half_size,args.top_y3+half_size,args.top_x4+half_size,args.top_y4+half_size,args.partition_color,half_size);
    	TaskLauncher Fourth_laucnher(B_LEGION_TASK_ID,TaskArgument(&Fourthargs,sizeof(Argument)));
    	Fourth_laucnher.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	Fourth_laucnher.add_region_requirement(RegionRequirement(fourthQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	Fourth_laucnher.add_field(0,FID_X);
    	Fourth_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,Fourth_laucnher);
	}

}

void c_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalRegion Alr = regions[1].get_logical_region();
    int size = args.size;
    int half_size = size/2;
    if(size <= legion_threshold) {
    	TaskLauncher C_Serial(C_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	C_Serial.add_region_requirement(RegionRequirement(lr,READ_WRITE,EXCLUSIVE,lr));
    	C_Serial.add_region_requirement(RegionRequirement(Alr,READ_ONLY,EXCLUSIVE,Alr));
    	C_Serial.add_field(0,FID_X);
    	C_Serial.add_field(1,FID_X);
    	runtime->execute_task(ctx,C_Serial);
    }
    else{
    	int tx1 = args.top_x1;
    	int ty1 = args.top_y1;
    	int tx2 = args.top_x2;
    	int ty2 = args.top_y2;
    	DomainPointColoring coloring;
    	IndexSpace is = lr.get_index_space();
    	int add = half_size-1;
    	LogicalPartition lp;
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx1, ty1), make_point(tx1+add, ty1+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx1, ty1+half_size), make_point(tx1+add, ty1+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx1+half_size, ty1), make_point(tx1+half_size+add, ty1+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx1+half_size, ty1+half_size), make_point(tx1+half_size+add, ty1+half_size+add)));
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
    		coloring[0] = Domain(Rect<2>(make_point(tx2, ty2), make_point(tx2+add, ty2+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx2, ty2+half_size), make_point(tx2+add, ty2+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx2+half_size, ty2), make_point(tx2+half_size+add, ty2+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx2+half_size, ty2+half_size), make_point(tx2+half_size+add, ty2+half_size+add)));
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


    	Argument Firstargs(tx1,ty1,tx2,ty2,args.top_x3,args.top_y3,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher First_launcher(C_LEGION_TASK_ID, TaskArgument(&Firstargs,sizeof(Argument)));
    	First_launcher.add_region_requirement(RegionRequirement(firstQuad,READ_WRITE,EXCLUSIVE,lr));
    	First_launcher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	First_launcher.add_field(0,FID_X);
    	First_launcher.add_field(1,FID_X);
    	runtime->execute_task(ctx,First_launcher);

    	Argument Secondargs(tx1+half_size,ty1,tx2+half_size,ty2,args.top_x3,args.top_y3,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher Second_laucnher(C_LEGION_TASK_ID,TaskArgument(&Secondargs,sizeof(Argument)));
    	Second_laucnher.add_region_requirement(RegionRequirement(thirdQuad,READ_WRITE,EXCLUSIVE,lr));
    	Second_laucnher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	Second_laucnher.add_field(0,FID_X);
    	Second_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,Second_laucnher);

    	Argument Dargs1(tx1,ty1+half_size,tx2,ty2,args.top_x3,args.top_y3+half_size,args.top_x4,args.top_y4,args.partition_color,half_size);
    	TaskLauncher D_laucnher(D_LEGION_TASK_ID,TaskArgument(&Dargs1,sizeof(Argument)));
    	D_laucnher.add_region_requirement(RegionRequirement(secondQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_laucnher.add_region_requirement(RegionRequirement(secondQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher.add_field(0,FID_X);
    	D_laucnher.add_field(1,FID_X);
    	D_laucnher.add_field(2,FID_X);
    	D_laucnher.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher);

		Argument Dargs2(tx1+half_size,ty1+half_size,tx2+half_size,ty2,args.top_x3,args.top_y3+half_size,args.top_x4,args.top_y4,args.partition_color,half_size);    	
    	TaskLauncher D_laucnher2(D_LEGION_TASK_ID,TaskArgument(&Dargs2,sizeof(Argument)));
    	D_laucnher2.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher2.add_region_requirement(RegionRequirement(thirdQuad,READ_ONLY,EXCLUSIVE,lr));
    	D_laucnher2.add_region_requirement(RegionRequirement(secondQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher2.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	D_laucnher2.add_field(0,FID_X);
    	D_laucnher2.add_field(1,FID_X);
    	D_laucnher2.add_field(2,FID_X);
    	D_laucnher2.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher2);

    	Argument Thirdargs(tx1,ty1+half_size,tx2,ty2+half_size,args.top_x3+half_size,args.top_y3+half_size,args.top_x4+half_size,args.top_y4+half_size,args.partition_color,half_size);
    	TaskLauncher Third_launcher(C_LEGION_TASK_ID, TaskArgument(&Thirdargs,sizeof(Argument)));
    	Third_launcher.add_region_requirement(RegionRequirement(secondQuad,READ_WRITE,EXCLUSIVE,lr));
    	Third_launcher.add_region_requirement(RegionRequirement(fourthQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	Third_launcher.add_field(0,FID_X);
    	Third_launcher.add_field(1,FID_X);
    	runtime->execute_task(ctx,Third_launcher);

    	Argument Fourthargs(tx1+half_size,ty1+half_size,tx2+half_size,ty2+half_size,args.top_x3+half_size,args.top_y3+half_size,args.top_x4+half_size,args.top_y4+half_size,args.partition_color,half_size);
    	TaskLauncher Fourth_laucnher(C_LEGION_TASK_ID,TaskArgument(&Fourthargs,sizeof(Argument)));
    	Fourth_laucnher.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	Fourth_laucnher.add_region_requirement(RegionRequirement(fourthQuadA,READ_ONLY,EXCLUSIVE,Alr));
    	Fourth_laucnher.add_field(0,FID_X);
    	Fourth_laucnher.add_field(1,FID_X);
    	runtime->execute_task(ctx,Fourth_laucnher);
	}

}

void d_legion_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalRegion secondlr = regions[1].get_logical_region();
    LogicalRegion thirdlr = regions[2].get_logical_region();
    LogicalRegion fourthlr = regions[3].get_logical_region();
    int size = args.size;
    int half_size = size/2;

    if(size <= legion_threshold) {
    	TaskLauncher D_Serial(D_NON_LEGION_TASK_ID, TaskArgument(&args,sizeof(Argument)));
    	D_Serial.add_region_requirement(RegionRequirement(lr,READ_WRITE,EXCLUSIVE,lr));
    	D_Serial.add_region_requirement(RegionRequirement(secondlr,READ_ONLY,EXCLUSIVE,secondlr));
    	D_Serial.add_region_requirement(RegionRequirement(thirdlr,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_Serial.add_region_requirement(RegionRequirement(fourthlr,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_Serial.add_field(0,FID_X);
    	D_Serial.add_field(1,FID_X);
    	D_Serial.add_field(2,FID_X);
    	D_Serial.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_Serial);
    }
    else{
    	int tx1 = args.top_x1;
    	int ty1 = args.top_y1;
    	int tx2 = args.top_x2;
    	int ty2 = args.top_y2;
    	int tx3 = args.top_x3;
    	int ty3 = args.top_y3;
    	int tx4 = args.top_x4;
    	int ty4 = args.top_y4;
    	DomainPointColoring coloring;
    	IndexSpace is = lr.get_index_space();
    	int add = half_size-1;
    	LogicalPartition lp;
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx1, ty1), make_point(tx1+add, ty1+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx1, ty1+half_size), make_point(tx1+add, ty1+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx1+half_size, ty1), make_point(tx1+half_size+add, ty1+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx1+half_size, ty1+half_size), make_point(tx1+half_size+add, ty1+half_size+add)));
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

    	is = secondlr.get_index_space();
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx2, ty2), make_point(tx2+add, ty2+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx2, ty2+half_size), make_point(tx2+add, ty2+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx2+half_size, ty2), make_point(tx2+half_size+add, ty2+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx2+half_size, ty2+half_size), make_point(tx2+half_size+add, ty2+half_size+add)));
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


    	is = thirdlr.get_index_space();
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx3, ty3), make_point(tx3+add, ty3+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx3, ty3+half_size), make_point(tx3+add, ty3+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx3+half_size, ty3), make_point(tx3+half_size+add, ty3+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx3+half_size, ty3+half_size), make_point(tx3+half_size+add, ty3+half_size+add)));
    		Rect<1>color_space = Rect<1>(0,3);
    		IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    		lp = runtime->get_logical_partition(ctx, lr, ip);
    	}
    	else{
    		 lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
    	}
    	LogicalRegion firstQuadB = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    	LogicalRegion secondQuadB = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    	LogicalRegion thirdQuadB = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    	LogicalRegion fourthQuadB = runtime->get_logical_subregion_by_color(ctx, lp, 3);


    	is = fourthlr.get_index_space();
    	if(!runtime->has_index_partition(ctx,is,args.partition_color)){
    		coloring[0] = Domain(Rect<2>(make_point(tx3, ty3), make_point(tx3+add, ty3+add)));
    		coloring[1] = Domain(Rect<2>(make_point(tx3, ty3+half_size), make_point(tx3+add, ty3+half_size+add)));
    		coloring[2] = Domain(Rect<2>(make_point(tx3+half_size, ty3), make_point(tx3+half_size+add, ty3+add)));
    		coloring[3] = Domain(Rect<2>(make_point(tx3+half_size, ty3+half_size), make_point(tx3+half_size+add, ty3+half_size+add)));
    		Rect<1>color_space = Rect<1>(0,3);
    		IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
    		lp = runtime->get_logical_partition(ctx, lr, ip);
    	}
    	else{
    		 lp = runtime->get_logical_partition_by_color(ctx, lr, args.partition_color);
    	}
    	LogicalRegion firstQuadC = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    	LogicalRegion secondQuadC = runtime->get_logical_subregion_by_color(ctx, lp, 1);
    	LogicalRegion thirdQuadC = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    	LogicalRegion fourthQuadC = runtime->get_logical_subregion_by_color(ctx, lp, 3);

    	Argument Dargs1(tx1,ty1,tx2,ty2,tx3,ty3,tx4,ty4,args.partition_color,half_size);
    	TaskLauncher D_laucnher(D_LEGION_TASK_ID,TaskArgument(&Dargs1,sizeof(Argument)));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher.add_region_requirement(RegionRequirement(firstQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher.add_field(0,FID_X);
    	D_laucnher.add_field(1,FID_X);
    	D_laucnher.add_field(2,FID_X);
    	D_laucnher.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher);

    	Argument Dargs2(tx1,ty1+half_size,tx2,ty2,tx3,ty3+half_size,tx4,ty4,args.partition_color,half_size);
    	TaskLauncher D_laucnher2(D_LEGION_TASK_ID,TaskArgument(&Dargs2,sizeof(Argument)));
    	D_laucnher2.add_region_requirement(RegionRequirement(secondQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher2.add_region_requirement(RegionRequirement(firstQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher2.add_region_requirement(RegionRequirement(secondQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher2.add_region_requirement(RegionRequirement(firstQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher2.add_field(0,FID_X);
    	D_laucnher2.add_field(1,FID_X);
    	D_laucnher2.add_field(2,FID_X);
    	D_laucnher2.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher2);

    	Argument Dargs3(tx1+half_size,ty1,tx2+half_size,ty2,tx3,ty3,tx4,ty4,args.partition_color,half_size);
    	TaskLauncher D_laucnher3(D_LEGION_TASK_ID,TaskArgument(&Dargs3,sizeof(Argument)));
    	D_laucnher3.add_region_requirement(RegionRequirement(thirdQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher3.add_region_requirement(RegionRequirement(thirdQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher3.add_region_requirement(RegionRequirement(firstQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher3.add_region_requirement(RegionRequirement(firstQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher3.add_field(0,FID_X);
    	D_laucnher3.add_field(1,FID_X);
    	D_laucnher3.add_field(2,FID_X);
    	D_laucnher3.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher3);

    	Argument Dargs4(tx1+half_size,ty1+half_size,tx2+half_size,ty2,tx3,ty3+half_size,tx4,ty4,args.partition_color,half_size);
    	TaskLauncher D_laucnher4(D_LEGION_TASK_ID,TaskArgument(&Dargs4,sizeof(Argument)));
    	D_laucnher4.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher4.add_region_requirement(RegionRequirement(thirdQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher4.add_region_requirement(RegionRequirement(secondQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher4.add_region_requirement(RegionRequirement(firstQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher4.add_field(0,FID_X);
    	D_laucnher4.add_field(1,FID_X);
    	D_laucnher4.add_field(2,FID_X);
    	D_laucnher4.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher4);

    	Argument Dargs5(tx1,ty1,tx2,ty2+half_size,tx3+half_size,ty3,tx4+half_size,ty4+half_size,args.partition_color,half_size);
    	TaskLauncher D_laucnher5(D_LEGION_TASK_ID,TaskArgument(&Dargs5,sizeof(Argument)));
    	D_laucnher5.add_region_requirement(RegionRequirement(firstQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher5.add_region_requirement(RegionRequirement(secondQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher5.add_region_requirement(RegionRequirement(thirdQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher5.add_region_requirement(RegionRequirement(fourthQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher5.add_field(0,FID_X);
    	D_laucnher5.add_field(1,FID_X);
    	D_laucnher5.add_field(2,FID_X);
    	D_laucnher5.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher5);

    	Argument Dargs6(tx1,ty1+half_size,tx2,ty2+half_size,tx3+half_size,ty3+half_size,tx4+half_size,ty4+half_size,args.partition_color,half_size);
    	TaskLauncher D_laucnher6(D_LEGION_TASK_ID,TaskArgument(&Dargs6,sizeof(Argument)));
    	D_laucnher6.add_region_requirement(RegionRequirement(secondQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher6.add_region_requirement(RegionRequirement(secondQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher6.add_region_requirement(RegionRequirement(fourthQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher6.add_region_requirement(RegionRequirement(fourthQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher6.add_field(0,FID_X);
    	D_laucnher6.add_field(1,FID_X);
    	D_laucnher6.add_field(2,FID_X);
    	D_laucnher6.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher6);

    	Argument Dargs7(tx1+half_size,ty1,tx2+half_size,ty2+half_size,tx3+half_size,ty3,tx4+half_size,ty4+half_size,args.partition_color,half_size);
    	TaskLauncher D_laucnher7(D_LEGION_TASK_ID,TaskArgument(&Dargs7,sizeof(Argument)));
    	D_laucnher7.add_region_requirement(RegionRequirement(thirdQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher7.add_region_requirement(RegionRequirement(fourthQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher7.add_region_requirement(RegionRequirement(thirdQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher7.add_region_requirement(RegionRequirement(fourthQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher7.add_field(0,FID_X);
    	D_laucnher7.add_field(1,FID_X);
    	D_laucnher7.add_field(2,FID_X);
    	D_laucnher7.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher7);

    	Argument Dargs8(tx1+half_size,ty1+half_size,tx2+half_size,ty2+half_size,tx3+half_size,ty3+half_size,tx4+half_size,ty4+half_size,args.partition_color,half_size);
    	TaskLauncher D_laucnher8(D_LEGION_TASK_ID,TaskArgument(&Dargs8,sizeof(Argument)));
    	D_laucnher8.add_region_requirement(RegionRequirement(fourthQuad,READ_WRITE,EXCLUSIVE,lr));
    	D_laucnher8.add_region_requirement(RegionRequirement(fourthQuadA,READ_ONLY,EXCLUSIVE,secondlr));
    	D_laucnher8.add_region_requirement(RegionRequirement(fourthQuadB,READ_ONLY,EXCLUSIVE,thirdlr));
    	D_laucnher8.add_region_requirement(RegionRequirement(fourthQuadC,READ_ONLY,EXCLUSIVE,fourthlr));
    	D_laucnher8.add_field(0,FID_X);
    	D_laucnher8.add_field(1,FID_X);
    	D_laucnher8.add_field(2,FID_X);
    	D_laucnher8.add_field(3,FID_X);
    	runtime->execute_task(ctx,D_laucnher8);
	}

}


double convert(string s){
	if(s[0]==' ')
		s.erase(s.begin());
	return stod(s);
}

vector<double> parse(string s){
	s.erase(s.begin());
	s.pop_back();
	stringstream ss(s);
	string input;
	vector<double>vals;
	while(getline(ss,input,',')){
		vals.push_back(convert(input));
	}
	return vals;
}

void populate_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	SingleMat args = task->is_index_space ? *(const SingleMat *) task->local_args
    : *(const SingleMat *) task->args;
    const FieldAccessor<WRITE_DISCARD, double, 2> write_acc(regions[0], FID_X);
    freopen("Input.txt","r",stdin);
    string s;
    for(int i = args.top_x ; i <= args.bottom_x ; i++) {
    	getline(cin,s);
    	vector<double>input = parse(s);
    	for(int j = args.top_y ; j <= args.bottom_y ; j++ ){
    		write_acc[make_point(i,j)] = input[j];
    	}
    }
}

void print_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	SingleMat args = task->is_index_space ? *(const SingleMat *) task->local_args
    : *(const SingleMat *) task->args;
    const FieldAccessor<READ_ONLY, double, 2> read_acc(regions[0], FID_X);
    for(int i = args.top_x ; i <= args.bottom_x ; i++) {
    	for(int j = args.top_y ; j <= args.bottom_y ; j++ ){
    		cout<<read_acc[make_point(i,j)]<<" ";
    	} cout<<endl;
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
        TaskVariantRegistrar registrar(PRINT_TASK_ID, "print_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<print_task>(registrar, "print_task");
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
    {
        TaskVariantRegistrar registrar(A_NON_LEGION_TASK_ID, "a_non_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<a_non_legion_task>(registrar, "a_non_legion_task");
    }
    {
        TaskVariantRegistrar registrar(B_NON_LEGION_TASK_ID, "b_non_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<b_non_legion_task>(registrar, "b_non_legion_task");
    }
    {
        TaskVariantRegistrar registrar(C_NON_LEGION_TASK_ID, "c_non_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<c_non_legion_task>(registrar, "c_non_legion_task");
    }
    {
        TaskVariantRegistrar registrar(D_NON_LEGION_TASK_ID, "d_non_legion_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<d_non_legion_task>(registrar, "d_non_legion_task");
    }
    return Runtime::start(argc, argv);
}