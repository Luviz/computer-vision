# Retrospective on hand gesture recognizer

Not, bad for my first attempt to use Tensorflow.
Well now that I have completed my research on this specific subject,
I realize that the model configuration is just OVER KILL.

However, I have achieved a model that is quite accurate and fast to train on a CPU.

With all of that, using Media-Pipe made this entire project trivial.

## The Good

I was Able to create a very accurate with my model, on the few types that I have.
The Model is fast to train, it takes around 30 sec to train it to 100% accuracy with my current configurations.

I have learned alto about Media-Pipe, numpy and open cv. I have improved on my linear algebra.

I was able to get a good visualization on how the model would need to be constructed,
before building the model.
This fact proves that I have gained a firm understating on the subject of machine learning.

## The Bad

The model is too big and it could be reduced in size. the model structure needs to improve.
I need to add more gestures I thing, and that would create more conflict.

My conversation of landmarks to numpy array is slow enough,
to case major latency where there are more then one hands.
However it looks like using it directly in a for loop manges the speed issue.
In short we need create a better hand processing function.

## The Ugly

The Code structure is just horrible.
It mainly due to prototype style of code that I have written.
I believe that I can improve on that by creating a dedicated repository.
The folder structure, below should be improve code structure.

```
├─ modules
├─ model
└- main.py
```

The core Idea is to always start from `main.py` and call modules from modules folder.
The notebooks need to at root as well to alow use local imports.
