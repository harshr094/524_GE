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
    A_TASK_ID,
    B_TASK_ID,
    PRINT_TASK_ID,
    T_TASK_ID

};

enum FieldId{
    FID_X,
};

struct Argument
{
	    Color partition_color;
	    int l;
	    int r;
	    Argument(int _l, int _r, Color _partition){
	    	l = _l;
	    	r = _r;
	    	partition_color = _partition;
	    }
};

struct PrintArgument
{
	int l;
	int r;
	long long int cnt;
	string call;
	PrintArgument(int _l, int _r, int _cnt, string _call){
		l=_l;
		r=_r;
		cnt=_cnt;
		call = _call;
	}
};



void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {
	int m = 10;
	Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, m)));
	IndexSpace is = runtime->create_index_space(ctx, tree_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(int), FID_X);
    }
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Color partition_color1 = 10;
    Argument args(0,pow(2,m),partition_color1);
    TaskLauncher T_launcher(T_TASK_ID, TaskArgument(&args, sizeof(Argument)));
    T_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    T_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, T_launcher);
}

void print_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	PrintArgument args = task->is_index_space ? *(const PrintArgument *) task->local_args
    : *(const PrintArgument *) task->args;
    cout<<"Called by "<<args.call<<endl;
    if(args.call[0]=='A'){
    	    const FieldAccessor<WRITE_DISCARD, int, 1> write_acc(regions[0], FID_X);
    	    write_acc[args.l]=55;
    }
    else{
    	const FieldAccessor<READ_ONLY, int, 1> read_acc(regions[0], FID_X);
    }
    cout<<"Starting "<<args.call<<endl;
    for(int i = 0 ; i < args.cnt; i++);
   	cout<<"Ending "<<args.call<<endl;
}


void t_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    int l = args.l;
    int r = args.r;
    LogicalRegion lr = regions[0].get_logical_region();
    int m = (l+r)/2;
    DomainPointColoring colorStartTile;
    LogicalRegion leftHalf;
    LogicalRegion rightHalf;
    colorStartTile[0] = Rect<1>(l,m);
    colorStartTile[1] = Rect<1>(m+1,r);
    Rect<1>color_space = Rect<1>(0,1);
    IndexSpace is = lr.get_index_space();
    IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, colorStartTile, DISJOINT_KIND, args.partition_color);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    leftHalf = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    rightHalf = runtime->get_logical_subregion_by_color(ctx,lp,1);

    Argument aArgs(l,m,args.partition_color);
    TaskLauncher A_launcher(A_TASK_ID,TaskArgument(&aArgs,sizeof(Argument)));
    RegionRequirement req1(leftHalf, WRITE_DISCARD, EXCLUSIVE, lr);
    req1.add_field(FID_X);
    A_launcher.add_region_requirement(req1);
    runtime->execute_task(ctx,A_launcher);

    TaskLauncher B_launcher(B_TASK_ID,TaskArgument(&args,sizeof(Argument)));
    RegionRequirement req3(leftHalf,READ_ONLY,EXCLUSIVE,lr);
    req3.add_field(FID_X);
    RegionRequirement req2(rightHalf,WRITE_DISCARD,EXCLUSIVE,lr);
    req2.add_field(FID_X);
    B_launcher.add_region_requirement(req3);
    B_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,B_launcher);
}

void a_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    cout<<"Called by A"<<endl;
    int l = args.l;
    int r = args.r;
    LogicalRegion lr = regions[0].get_logical_region();
    int m = (l+r)/2;
    DomainPointColoring colorStartTile;
    colorStartTile[0] = Rect<1>(l,m);
    colorStartTile[1] = Rect<1>(m+1,r);
    Rect<1>color_space = Rect<1>(0,1);
    IndexSpace is = lr.get_index_space();
    IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, colorStartTile, DISJOINT_KIND, args.partition_color);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    LogicalRegion leftHalf = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    LogicalRegion rightHalf = runtime->get_logical_subregion_by_color(ctx,lp,1);

    PrintArgument PrintArgs(l,m,10,"A1");
    TaskLauncher Print_launcher(PRINT_TASK_ID,TaskArgument(&PrintArgs,sizeof(PrintArgument)));
    RegionRequirement req1(leftHalf, WRITE_DISCARD, EXCLUSIVE, lr);
    req1.add_field(FID_X);
    Print_launcher.add_region_requirement(req1);
    runtime->execute_task(ctx,Print_launcher);

    PrintArgument PrintArgs2(m+1,r,500000000,"A2");
    TaskLauncher Print_launcher2(PRINT_TASK_ID,TaskArgument(&PrintArgs2,sizeof(PrintArgument)));
    RegionRequirement req2(rightHalf, WRITE_DISCARD, EXCLUSIVE, lr);
    req2.add_field(FID_X);
    RegionRequirement req3(leftHalf,READ_ONLY,EXCLUSIVE,lr);
    req3.add_field(FID_X);
    Print_launcher2.add_region_requirement(req2);
    Print_launcher2.add_region_requirement(req3);
    runtime->execute_task(ctx,Print_launcher2);
    cout<<"Finished by A"<<endl;

}


void b_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
	Argument args = task->is_index_space ? *(const Argument *) task->local_args
    : *(const Argument *) task->args;
    cout<<"Called by B"<<endl;
	int l = args.l;
    int r = args.r;
    LogicalRegion lr = regions[1].get_logical_region();
    int m = (l+r)/2;
    int m1 = (m+r)/2;
    DomainPointColoring colorStartTile;
    colorStartTile[0] = Rect<1>(m,m1);
    colorStartTile[1] = Rect<1>(m1+1,r);
    Rect<1>color_space = Rect<1>(2, 3);
    IndexSpace is = lr.get_index_space();
    IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, colorStartTile, DISJOINT_KIND, args.partition_color);
    LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
    LogicalRegion bLeft = runtime->get_logical_subregion_by_color(ctx, lp, 2);
    LogicalRegion bRight = runtime->get_logical_subregion_by_color(ctx,lp,3);

    int m2 = (l+m)/2;
    LogicalRegion lr1 = regions[0].get_logical_region();
    LogicalPartition lp1 = runtime->get_logical_partition_by_color(ctx,lr1,args.partition_color);
    LogicalRegion leftHalf = runtime->get_logical_subregion_by_color(ctx, lp1, 0);
    LogicalRegion rightHalf = runtime->get_logical_subregion_by_color(ctx, lp1, 1);


    PrintArgument PrintArgs2(l,m2,10,"B1");
    TaskLauncher Print_launcher2(PRINT_TASK_ID,TaskArgument(&PrintArgs2,sizeof(PrintArgument)));
    RegionRequirement req3(rightHalf, READ_ONLY, EXCLUSIVE, lr1);
    req3.add_field(FID_X);
    RegionRequirement req4(bRight,WRITE_DISCARD,EXCLUSIVE,lr);
    req4.add_field(FID_X);
    Print_launcher2.add_region_requirement(req3);
    Print_launcher2.add_region_requirement(req4);
    runtime->execute_task(ctx,Print_launcher2);

    PrintArgument PrintArgs(m2+1,m,10,"B2");
    TaskLauncher Print_launcher(PRINT_TASK_ID,TaskArgument(&PrintArgs,sizeof(PrintArgument)));
    RegionRequirement req1(leftHalf, READ_ONLY, EXCLUSIVE, lr1);
    req1.add_field(FID_X);
    RegionRequirement req2(bLeft,WRITE_DISCARD,EXCLUSIVE,lr);
    req2.add_field(FID_X);
    Print_launcher.add_region_requirement(req1);
    Print_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,Print_launcher);
    cout<<"Finished by B"<<endl;
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
        TaskVariantRegistrar registrar(PRINT_TASK_ID, "print");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<print_task>(registrar, "print");
    }
    {
        TaskVariantRegistrar registrar(A_TASK_ID, "atask");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<a_task>(registrar, "atask");
    }
    {
        TaskVariantRegistrar registrar(B_TASK_ID, "btask");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<b_task>(registrar, "btask");
    }
    {
        TaskVariantRegistrar registrar(T_TASK_ID, "ttask");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<t_task>(registrar, "ttask");
    }
    return Runtime::start(argc, argv);
}