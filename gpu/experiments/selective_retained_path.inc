// Research-only replacement for classical selection, included inside the rules
// namespace after the unmodified reference template. All scores/open metadata
// stay fixed during choose. An eligible retained prefix remains canonical under
// monotone reservations; only fully reserved suffixes need rediscovery.
__device__ void selective_choose_retained_path(const Node *nodes,const uint32_t *counts,const SelectMeta *meta,
    uint32_t *reserved,SelectWork *work,uint64_t used,uint64_t capacity,uint64_t remaining,
    uint32_t batch,uint32_t candidate_limit,uint32_t max_ply,uint32_t *error) {
    static_assert(CHEN_SELECTIVE_MAX_PLY<=CHEN_SELECTIVE_MAX_BATCH,"path fits existing backup scratch");
    SelectHead head{0,0,0,CHEN_SELECTIVE_PROGRESS};
    bool budget_rejected=false,capacity_rejected=false;
    if(!meta[0].open) { head.stop=CHEN_SELECTIVE_CLOSED;work->head=head;return; }
    if(!remaining) { head.stop=CHEN_SELECTIVE_NODE_BUDGET;work->head=head;return; }
    if(used==capacity) { head.stop=CHEN_SELECTIVE_ARENA_CAPACITY;work->head=head;return; }
    if(!max_ply || max_ply>CHEN_SELECTIVE_MAX_PLY || nodes[0].parent!=NONE) { device_error(error,4);return; }
    // Backup rebuilds these buckets only after choose and expansion finish.
    // No new allocation, layout, clearing pass or stack scratch is introduced.
    uint64_t *path=work->sparse.nodes[0];
    uint32_t path_size=1;path[0]=0;
    while(head.examined<candidate_limit && head.selected<batch &&
        meta[0].open>selective_reservations<true>(reserved[0])) {
        uint64_t index=path[path_size-1];
        while(nodes[index].flags&EXPANDED) {
            const uint64_t next=selective_open_child<true>(nodes,meta,reserved,index,used,error);
            if(next==NONE) { device_error(error,4);return; }
            if(path_size>=max_ply || nodes[next].parent!=index || nodes[next].ply!=path_size) {
                device_error(error,4);return;
            }
            path[path_size++]=next;index=next;
        }
        const Node leaf=nodes[index];
        if(!(leaf.flags&CLASSIFIED) || (leaf.flags&TERMINAL) || leaf.ply>=max_ply ||
            (index && !(leaf.flags&SCORED)) || !counts[index] || selective_reservations<true>(reserved[index])) {
            device_error(error,4);return;
        }
        ++head.examined;
        for(uint32_t cursor=path_size;cursor;) {
            const uint64_t parent=path[--cursor];
            if(parent>=used || selective_reservations<true>(reserved[parent])>=candidate_limit) {
                device_error(error,4);return;
            }
            selective_reserve<true>(reserved[parent]);
        }
        const uint64_t cost=counts[index];
        const bool node_fits=cost<=remaining-head.added;
        const bool arena_fits=cost<=capacity-used-head.added;
        budget_rejected |= !node_fits;capacity_rejected |= !arena_fits;
        if(node_fits && arena_fits) {
            work->selected[head.selected]=index;work->offsets[head.selected]=head.added;
            ++head.selected;head.added+=uint32_t(cost);
        }
        while(path_size>1) {
            const uint64_t tail=path[path_size-1];
            if(meta[tail].open>selective_reservations<true>(reserved[tail])) break;
            --path_size;
        }
    }
    if(!head.selected) {
        if(head.examined==candidate_limit && meta[0].open>selective_reservations<true>(reserved[0]))
            head.stop=CHEN_SELECTIVE_CANDIDATE_SCAN;
        else if(budget_rejected) head.stop=CHEN_SELECTIVE_NODE_BUDGET;
        else if(capacity_rejected) head.stop=CHEN_SELECTIVE_ARENA_CAPACITY;
        else { device_error(error,4);return; }
    }
    work->head=head;
}
template<bool Cached> __device__ void selective_choose_device(const Node *nodes,const uint32_t *counts,const SelectMeta *meta,
    uint32_t *reserved,SelectWork *work,uint64_t used,uint64_t capacity,uint64_t remaining,
    uint32_t batch,uint32_t candidate_limit,uint32_t max_ply,uint32_t *error) {
    if constexpr(Cached) selective_choose_retained_path(nodes,counts,meta,reserved,work,used,capacity,remaining,
        batch,candidate_limit,max_ply,error);
    else selective_choose_device_reference<false>(nodes,counts,meta,reserved,work,used,capacity,remaining,
        batch,candidate_limit,max_ply,error);
}
