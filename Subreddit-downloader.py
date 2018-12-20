#!/usr/bin/env python3

import praw
import json
import random

AMOUNT_OF_POSTS = 1000


reddit = praw.Reddit(client_id='Elgsv-j1KV13Cg',
                     client_secret='lIZOnlA0oBTxdO88Bknk5hSKfm8',
                     user_agent='windows:SubredditFinder:1.0 (by Jordan Garces)')


def dumpList(data, target, name):

    c = list(zip(data, target))
    random.shuffle(c)
    data, target = zip(*c)

    halfData = int(len(data)/2)
    threeQtrsData = int(len(data)/2 + len(data)/4)

    training = list()
    training.append(data[:halfData])
    training.append(target[:halfData])

    development = list()
    development.append(data[halfData:threeQtrsData])
    development.append(target[halfData:threeQtrsData])

    test = list()
    test.append(data[threeQtrsData:len(data)])
    test.append(target[threeQtrsData:len(data)])

    with open('training' + name + '.json', 'w') as fp:
        json.dump(training, fp)
    with open('development' + name + '.json', 'w') as fp:
        json.dump(development, fp)
    with open('testing' + name + '.json', 'w') as fp:
        json.dump(test, fp)


if __name__ == '__main__':

    data_5subs = list()
    target_5subs = list()
    data_2subs = list()
    target_2subs = list()

    print('Downloading from /r/WritingPrompts')
    for submission in reddit.subreddit('WritingPrompts').top(limit=AMOUNT_OF_POSTS, time_filter='all'):
        data_5subs.append(submission.selftext)
        target_5subs.append(0)

    print('Downloading from /r/truegaming')
    for submission in reddit.subreddit('truegaming').top(limit=AMOUNT_OF_POSTS, time_filter='all'):
        data_5subs.append(submission.selftext)
        target_5subs.append(1)
        data_2subs.append(submission.selftext)
        target_2subs.append(0)

    print('Downloading from /r/summonerschool')
    for submission in reddit.subreddit('summonerschool').top(limit=AMOUNT_OF_POSTS, time_filter='all'):
        data_5subs.append(submission.selftext)
        target_5subs.append(2)

    print('Downloading from /r/lifeofnorman')
    for submission in reddit.subreddit('lifeofnorman').top(limit=AMOUNT_OF_POSTS, time_filter='all'):
        data_5subs.append(submission.selftext)
        target_5subs.append(3)

    print('Downloading from /r/nosleep')
    for submission in reddit.subreddit('nosleep').top(limit=AMOUNT_OF_POSTS, time_filter='all'):
        data_5subs.append(submission.selftext)
        target_5subs.append(4)
        data_2subs.append(submission.selftext)
        target_2subs.append(1)

    dumpList(data_5subs, target_5subs, '-5subs')
    dumpList(data_2subs, target_2subs, '-2subs')
