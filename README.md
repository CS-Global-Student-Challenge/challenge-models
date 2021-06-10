# Challenge 1.1: Computer System Failure Data Analysis (Model Complete)
With the growing scale of supercomputing systems , scientists are now able to solve challenging computing problems in a matter of seconds which would take hundreds of years on a personal computer. However, with increasing scale (and complexity thereof) grows the probability of application failure, either due to hardware or software errors. Such application failures not only delay scientific progress, but also leads to a tremendous amount of wasted resources, both in terms of time and energy consumption. If we are able to predict when an application would fail due to a system error or due to software bugs, preventive mechanisms such as checkpointing can be initiated to save intermediate results, thereby, reducing the amount of wasted computation.

This assignment deals with predicting failure of application executions (referred to as “job”) on Purdue ITaP’s central computing cluster. This is data that we have collected, collated, and analyzed as part of two projects from the National Science Foundation (NSF), one completed (“Computer System Failure Data Repository to Enable Data-Driven Dependability Research”, Project No. CNS-1513197) and one ongoing (“Open Computer System Usage Repository and Analytics Engine”, Project No. CNS-2016704).

For each job, we have data about the resources the job uses and whether the job succeeded or failed. The resources for which we have data are:

* Memory
* Network
* Local IO
* Network File System (NFS)

We are releasing the training data from 20,000 jobs, which has about 8% failure data (this is referred to as the “positive class”). You will build Machine Learning models in Python to predict whether a job will fail or not, given the resource usage data. More details about the training and the test data are presented in the "Data" section.

The assignment, including the data, is available at the following Github repo:
https://github.com/bagchi/application-failure-prediction

There are two parts for this challenge problem with slightly different objectives:
Challenge 1.1 and Challenge 1.2. You must complete both parts and upload your solutions separately.

You will do the following steps for this Challenge Problem 1.1:

1. You will write code in Python. We have found the following packages to be useful for this task: pandas, sklearn, numpy.
2. You will use the training dataset train_data.csv.
3. You will create a Machine Learning model that achieves the highest balanced accuracy. Use all the features available in the dataset to train your model. We will call this “Model Complete”. You can use any algorithm of your choice.
4. Give a short writeup explaining:
  * What did you do for the Model Complete? Give the confusion matrix and the balanced accuracy that you obtained on the training data.
  * Upload this writeup.
5. Upload your Python code with a README that tells us how we can run your code.
