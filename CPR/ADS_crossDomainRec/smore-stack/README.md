## Compilation
```
make
```

## EdgeList.tsv as Input
```
nodeA   nodeB   5
nodeB   nodeC   3
nodeC   nodeA   4
```

## Usage
directly run the execution file to see the usage, for instance:
```
./miso
```

## Representation.tsv as Outputs
```
nodeA   0.123 0.321 0.456 0.765
nodeB   0.100 0.270 0.500 0.200
nodeC   0.174 0.124 0.190 0.147
```

## Take Medium Data as Example
given *user-item.tsv* like
```
[user]ffa0f90b10dc      [item]edb1de37d6d6      1.000000
[user]ffa0f90b10dc      [item]6127d236e5df      1.000000
[user]ffa0f90b10dc      [item]916669794ef1      1.000000
...
```
and *item-meta.tsv* like
```
[item]4ae4a30801a2      [tag]UX 1.000000
[item]d7df2527a9ed      [creatorId]b6886410781b 1.000000
[item]a5bc176714a3      [topic]Politics 1.000000
...
```
we can train the model by
```
./miso -train_ui user-item.tsv -train_iw item-meta.tsv -update_time 100 -save miso.embed
```
then you'll get output file named *miso.embed* 

Note. We usually set `-update_time` by the `number_of_edges*100/1000000`.

Finally you will receive *miso.embed* like
```
[user]ffa0f90b10dc  0.0815412 0.0205459 0.288714 0.296497 0.394043
[item]edb1de37d6d6  -0.207083 -0.258583 0.233185 0.0959801 0.258183
[tag]UX 0.0185886 0.138003 0.213609 0.276383 0.45732
[topic]Politics -0.0137994 -0.227462 0.103224 -0.456051 0.389858
...
```
Please contact CM (changecandy@gmail.com / cm.chen@askmiso.com) if you encounter any problem.
