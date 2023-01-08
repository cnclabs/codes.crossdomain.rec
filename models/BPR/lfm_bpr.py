import sys
import argparse
from scipy import sparse
from lightfm import LightFM

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--train', help='train file')
    parser.add_argument('--save', help='representation file')
    parser.add_argument('--dim', type=int, default=128, help='representation file')
    parser.add_argument('--iter', type=int, default=100, help='representation file')
    parser.add_argument('--worker', type=int, default=1, help='# of workers')
    parser.add_argument('--item_alpha', type=float, default=0.001, help='item_alpha')
    parser.add_argument('--user_alpha', type=float, default=0.001, help='item_alpha')
    args = parser.parse_args()
    
    model = LightFM(learning_rate=0.025, no_components=args.dim, loss='bpr', item_alpha=args.item_alpha, user_alpha=args.user_alpha)
    
    users = []
    items = []
    clicks = []
    user_map, user_index = {}, {}
    item_map, item_index = {}, {}
    
    sys.stderr.write('load train data ...\n')
    with open(args.train) as f:
        for line in f:
            user, item, click = line.rstrip('\n').split()
            if user not in user_index:
                user_index[user] = len(user_index)
                user_map[user_index[user]] = user
            if item not in item_index:
                item_index[item] = len(item_index)
                item_map[item_index[item]] = item
            users.append(user_index[user])
            items.append(item_index[item])
            clicks.append(float(click))
    train = sparse.coo_matrix((clicks, (users, items)))
    
    sys.stderr.write('start training ...\n')
    model.fit(train, epochs=args.iter, num_threads=args.worker, verbose=True)
    
    sys.stderr.write('save representations ...\n')
    user_reps = model.get_user_representations()
    item_reps = model.get_item_representations()
    
    rep_res = []
    # user amount, item amount
    # rep_res.append('%d %d' % (len(user_reps[1])+len(item_reps[1]), args.dim))
    # user_reps[0] : biases 
    # user_reps[1] : user_embeddings
    for e, user_rep in enumerate(user_reps[1]):
        rep_res.append('%s\t%s' % (user_map[e], ' '.join(list(map(str, user_rep)))))
    for e, item_rep in enumerate(item_reps[1]):
        if '_' in item_map[e]:
            item_map[e] = item_map[e][item_map[e].find("_")+1:]
        rep_res.append('%s\t%s' % (item_map[e], ' '.join(list(map(str, item_rep)))))
    
    print("save as: ", args.save)
    with open(args.save, 'w') as f:
        f.write('%s\n' % ('\n'.join(rep_res)))
