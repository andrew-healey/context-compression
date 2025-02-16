OK, here are some tests.
Well, how are we gonna verify that they acc ran?
Hrrrrrm. Ig we can just check the tags? Since theoretically our runs will finish the nohup before calling the subsequent commands.
Yeah let's do that actually. We can do like different duration runs. i.e. the first one finishes in 20 seconds, the next in 30 seconds, the next in 40 seconds, etc.
And we'll run these all on single-gpu systems.

```vast:finished
sleep 20
```

```vast:finished
sleep 30
```

```vast:finished
sleep 40
```
