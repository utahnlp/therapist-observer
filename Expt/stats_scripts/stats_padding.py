import sys, re

best_dev_loss = 1000.0
best_dev_loss_state = ''

dev_losses = {}
p_r_f1 = {}
recalls = {}
best_recall = {}
best_recall_state = {}
best_precision = {}
best_precision_state = {}
best_f1 = {}
best_f1_state = {}
support = {}

topK = [1,2,3,5,10]
for k in topK:
    best_recall[k] = -1.0

# keys = ['brownie','changetalk','sustain','neutral', 'reflection_complex','reflection_simple','affirm','facilitate','givinginfo','question_open','question_closed','advise_wp','advise_wop', 'confront', 'structure', 'other', 'avg / total']
keys = ['change_talk','sustain_talk','follow_neutral', 'facilitate','giving_info', 'reflection_simple','reflection_complex', 'question_closed','question_open', 'MI_adherent','MI_non-adherent', 'PADDING_LABEL', 'macro / total', 'micro / total', 'weighted_mac / total']
for key in keys:
    best_recall[key] = -1.0
    best_recall_state[key] = (0,0)
    best_precision[key] = -1.0
    best_precision_state[key] = (0,0)
    best_f1[key] = -1.0
    best_f1_state[key] = (0,0)
    p_r_f1[key] = {}

input_log = sys.argv[1]

for l in open(input_log):
    res = re.findall('Evaluating the model in the epoch (\d{1,}), after steps (\d{1,})', l)
    if res:
        epoch, steps = res[0]

    res3 = re.findall('Dev_eval_loss = (\d{1,}\.\d{1,})', l)
    if res3:
        dev_loss = float(res3[0])
        dev_losses[(epoch, steps)] = dev_loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dev_loss_state = (epoch, steps)

    #res4 = re.findall('recall@K = \[(.*)\]', l)
    #if res4:
    #    recalls[(epoch, steps)] = res4[0]
    #    for t in res4[0].split(', ('):
    #        p,r = t.replace('(', '').replace(')', '').split(', ')
    #        p = int(p)
    #        r = float(r)
    #        if r > best_recall[p]:
    #            best_recall[p] = r
    #            best_recall_state[p] = (epoch, steps)

    for key in keys:
        prf_res = re.findall('^\s*'+key+'\s+(\d{1,}\.\d{1,})\s+(\d{1,}\.\d{1,})\s+(\d{1,}\.\d{1,})\s+(\d{1,})', l)
        if prf_res:
            p, r, f1, s = prf_res[0]
            p_r_f1[key][(epoch, steps)]=(p,r,f1)
            support[key] = s
            if p >= best_precision[key]:
                best_precision[key] = p
                best_precision_state[key] = (epoch, steps)

            if r >= best_recall[key]:
                best_recall[key] = r
                best_recall_state[key] = (epoch, steps)

            if f1 >= best_f1[key]:
                best_f1[key] = f1
                best_f1_state[key] = (epoch, steps)


#for i in topK:
#    print "Best recall@%s: %s" % (i, best_recall[i])
#    epoch, steps = best_recall_state[i]
#    print "epoch: %s, %s, dev loss = %s" % (epoch, steps, dev_losses[best_recall_state[i]])
#    print "all recalls : %s" % (recalls[best_recall_state[i]])
#    print ''

print "Best dev loss: %s" % best_dev_loss
epoch, steps = best_dev_loss_state
print "epoch: %s, steps : %s" % (epoch, steps)
# print "recalls: %s" % recalls[best_dev_loss_state]
print ''

headers = ["best_precision", "best_recall", "best_f1-score"]
width = max(3, max([len(key) for key in keys]))
head_fmt = u'{:>{width}s} ' + u' {:>24}' * len(headers) + u'  support \n'
report = head_fmt.format(u'', *headers, width=width)
row_summary_fmt = u'{:>{width}s} ' + u' ({:>4}, {:>6}, {:>8})' * 3 + u'\n'
row_fmt = u'{:>{width}s} ' + u' ({:>4}, {:>6}, {:>8})' * 3 + u' {:>5}\n'
for key in keys:
    pepoch, psteps = best_precision_state[key]
    repoch, rsteps = best_recall_state[key]
    f1epoch, f1steps = best_f1_state[key]
    row1 = (key, best_precision[key], pepoch, psteps, best_recall[key], repoch, rsteps, best_f1[key], f1epoch, f1steps)
    report += row_summary_fmt.format(*row1, width = width)
    row2 = ('', p_r_f1[key][(pepoch, psteps)][0], p_r_f1[key][(pepoch, psteps)][1], p_r_f1[key][(pepoch, psteps)][2], p_r_f1[key][(repoch, rsteps)][0], p_r_f1[key][(repoch, rsteps)][1], p_r_f1[key][(repoch, rsteps)][2], p_r_f1[key][(f1epoch, f1steps)][0], p_r_f1[key][(f1epoch, f1steps)][1], p_r_f1[key][(f1epoch, f1steps)][2], support[key])
    report += row_fmt.format(*row2, width = width)

print(report)
