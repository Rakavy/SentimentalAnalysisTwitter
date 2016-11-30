import tweepy
import ujson

consumer_key = "SUAn5eS9GsAD3lLSDdyPp8Hc0"
consumer_secret="8U0ljImrgvZDMdYgI2RbZF1WXcDPsgb16YN1fmQIKSqwkafp0s"

access_token="1428703016-TalfNOsCLbsxkGXqVxp2nsnrdipYZkmR8OHYeAk"
access_token_secret="ZzTyCTEQhlzMDWAdumdU2hlYfnMZx2yrHsj26x10PrlYr"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

def twitterApi(searchQuery):
    maxTweets = 500 # Some arbitrary large number
    tweetsPerQry = 100  # this is the max the API permits
    fName = 'tweets.txt' # We'll store the tweets in a text file.
    # If results from a specific ID onwards are reqd, set since_id to that ID.
    # else default to no lower limit, go as far back as API allows
    sinceId = None

    # If results only below a specific ID are, set max_id to that ID.
    # else default to no upper limit, start from the most recent tweet matching the search query.
    max_id = -1

    tweetCount = 0
    print("Downloading max {0} tweets".format(maxTweets))
    with open(fName, 'w') as f:
        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    f.write(ujson.dumps(tweet._json) +
                            '\n')
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id
            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break

    print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

