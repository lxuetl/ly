"""In-sample inference-rule crossover; no retraining or test-set use."""
import hashlib,json,time
import torch
import torch.nn.functional as F
import diagnose_selector_fit32 as fit
tr=fit.tr

def main():
    torch.set_num_threads(4)
    out=fit.OUT/'factorial'
    out.mkdir(exist_ok=True)
    protocol=json.loads((fit.OUT/'protocol.json').read_text())
    data=[]
    for p in protocol['paths']:
        key=hashlib.sha256(p.encode()).hexdigest()[:24]
        item=torch.load(fit.SOURCE/'cache'/(key+'.pt'),map_location='cpu',weights_only=False)
        assert item['path']==p
        data.append(item)
    original=tr.TemporalSelector().cuda()
    utility=tr.TemporalSelector().cuda()
    for kind,m in [('original',original),('utility',utility)]:
        m.load_state_dict(torch.load(fit.OUT/(kind+'_last.pt'),map_location='cpu',weights_only=False)['model'])
        m.eval()
    def predict(m,b,kind):
        if kind.startswith('original'):
            w=m(b['features'])[0]
            if kind.endswith('hard'):
                w=F.one_hot(w.argmax(2),8).permute(0,1,4,2,3).float()
            return w
        if kind.endswith('hard'):return saved_predict(m,b,'utility')
        scores=m.net(b['features'].permute(0,2,1,3,4)).permute(0,2,1,3,4)[:,:,:8]
        w=scores.softmax(2) # fixed temperature 1 in original utility units
        gain=scores.amax(2)-scores[:,:,1]
        ref=torch.zeros_like(w);ref[:,:,1]=1
        return torch.where((gain>.1)[:,:,None],w,ref)
    saved_predict=fit.predict
    fit.predict=predict
    import lpips
    metric=lpips.LPIPS(net='alex',spatial=False,verbose=False).cuda().eval().requires_grad_(False)
    models={'original_soft':original,'original_hard':original,'utility_soft':utility,'utility_hard':utility}
    start=time.time()
    metrics,routing,rows=fit.evaluate(models,data,metric)
    result={'in_sample_only':True,'retrained':False,'clips':len(data),'metrics':metrics,'routing':routing,
        'protocol':'Same saved epoch80 checkpoints. Original hard=argmax of final gated weights. Utility soft=softmax(scores/1), same gain>0.1 fallback as utility hard. Same renderer/budget. This is inference crossover, not fully isolated training-objective factorial.',
        'seconds':time.time()-start}
    tr.prev.dump(out/'summary.json',result)
    (out/'per_clip.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    tr.prev.dump(out/'COMPLETE.json',{'seconds':time.time()-start})
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
