import edu.umn.cs.recsys.*
import edu.umn.cs.recsys.dao.*
import org.apache.commons.math3.distribution.GammaDistribution
import org.apache.commons.math3.distribution.UniformIntegerDistribution
import org.grouplens.lenskit.*
import org.grouplens.lenskit.baseline.*
import org.grouplens.lenskit.data.dao.ItemDAO
import org.grouplens.lenskit.data.dao.MapItemNameDAO
import org.grouplens.lenskit.data.dao.UserEventDAO
import org.grouplens.lenskit.data.text.CSVFileItemNameDAOProvider
import org.grouplens.lenskit.data.text.ItemFile
import org.grouplens.lenskit.eval.data.crossfold.RandomOrder
import org.grouplens.lenskit.eval.metrics.predict.*
import org.grouplens.lenskit.eval.metrics.topn.*
import org.grouplens.lenskit.mf.funksvd.*
import org.grouplens.lenskit.iterative.*

// common configuration to make tags available
// needed for both some algorithms and for metrics
// this defines a variable containing a Groovy closure, if you care about that kind of thing
tagConfig = {
    // use our tag data
    bind ItemTagDAO to CSVItemTagDAO
    // and our movie titles
    bind MapItemNameDAO toProvider CSVFileItemNameDAOProvider
    // configure input files for both of those
    set TagFile to new File("data/movie-tags.csv")
    set ItemFile to new File("data/movie-titles.csv")
    // need tag vocab & DAO to be roots for diversity metric to use them
    config.addRoot ItemTagDAO
    config.addRoot ItemDAO
    config.addRoot TagVocabulary
    config.addRoot UserEventDAO
}

// Run a train-test evaluation
trainTest {
    dataset crossfold("ml-100k") {
        source csvfile("ml-100k/u.data") {
//            file "data/ratings.csv"
//            headerLines 1
	    delimiter "\t"
            domain {
                minimum 0.5
                maximum 5
                precision 0.5
            }
        }
        train "build/ml-100k-1pct-users.%d.train.pack"
        test "build/ml-100k-1pct-users.%d.test.pack"

        // hold out 5 random items from each user
        order RandomOrder
        holdout 5

        // split users into 5 sets
        partitions 5
    }

    dataset crossfold("ml-1M") {
        source csvfile {
            file "ml-1M/ratings.dat"
            headerLines 1
            domain {
                minimum 0.5
                maximum 5
                precision 0.5
            }
        }
        train "build/ml-1m-1pct-users.%d.train.pack"
        test "build/ml-1m-1pct-users.%d.test.pack"

        // hold out 5 random items from each user
        order RandomOrder
        holdout 5

        // split users into 5 sets
        partitions 5
    }

    // Output aggregate results to eval-results.csv
    output "build/eval-results.csv"
    // Output per-user results to eval-user.csv
    userOutput "build/eval-user.csv"

    metric CoveragePredictMetric
    metric RMSEPredictMetric
    metric NDCGPredictMetric

    GammaDistribution meanGamma = new GammaDistribution(10.00,2.00)
    UniformIntegerDistribution integerDistribution = new UniformIntegerDistribution(5,150)
    GammaDistribution learnRate = new GammaDistribution(0.0005,2.00)
    GammaDistribution regularTerm = new GammaDistribution(0.001,2.00)

    // Compute nDCG trying to recommend lists of 10 from all items
    // This suffers from similar problems as the unary ratings case!

	metric topNnDCG {
        // candidates ItemSelectors.addNRandom(ItemSelectors.testItems(), 100)
	candidates ItemSelectors.allItems()
        exclude ItemSelectors.trainingItems()
        listSize 25
    }

	metric topNMRR {
        candidates ItemSelectors.addNRandom(ItemSelectors.testItems(), 100)
        goodItems ItemSelectors.testItems()
        listSize 25
    }

    metric topNMRR {
        suffix "HighRating"
        candidates ItemSelectors.addNRandom(ItemSelectors.testItems(), 100)
        goodItems ItemSelectors.testRatingMatches(org.hamcrest.Matchers.greaterThanOrEqualTo(3.5d))
        listSize 25
    }

    metric topNMAP {
        candidates ItemSelectors.addNRandom(ItemSelectors.testItems(), 100)
        goodItems ItemSelectors.testItems()
        listSize 25
    }
    metric topNMAP {
        suffix "HighRating"
        candidates ItemSelectors.addNRandom(ItemSelectors.testItems(), 100)
        goodItems ItemSelectors.testRatingMatches(org.hamcrest.Matchers.greaterThanOrEqualTo(3.5d))
        listSize 25
    }
	
    // measure the entropy of the top 10 items
    metric new TagEntropyMetric(10)

    for(int i=0;i<100;i++) {
        algorithm("FunkSVD") {
            double meanGammaSample = meanGamma.sample()
            int featureCountSample = integerDistribution.sample()
            double learnRateSample = learnRate.sample()
            double regularTermSample = regularTerm.sample()
            attributes["MeanDamping"] = meanGammaSample
            attributes["FeatureCount"] = featureCountSample
            attributes["LearningRate"] = learnRateSample
            attributes["RegularizationTerm"] = regularTermSample
            include tagConfig
            bind ItemScorer to FunkSVDItemScorer
            bind(BaselineScorer, ItemScorer) to UserMeanItemScorer
            bind(UserMeanBaseline, ItemScorer) to ItemMeanRatingItemScorer
            set MeanDamping to meanGammaSample
            set FeatureCount to featureCountSample
            set IterationCount to 125
            set LearningRate to learnRateSample
            set RegularizationTerm to regularTermSample
        }
    }
    algorithm("FunkSVDBasic") {
        double meanGammaSample = meanGamma.sample()
        int featureCountSample = integerDistribution.sample()
        double learnRateSample = learnRate.sample()
        double regularTermSample = regularTerm.sample()
        attributes["MeanDamping"] = meanGammaSample
        attributes["FeatureCount"] = featureCountSample
        attributes["LearningRate"] = learnRateSample
        attributes["RegularizationTerm"] = regularTermSample
        include tagConfig
        bind ItemScorer to FunkSVDItemScorer
        bind(BaselineScorer, ItemScorer) to UserMeanItemScorer
        bind(UserMeanBaseline, ItemScorer) to ItemMeanRatingItemScorer
        set MeanDamping to meanGammaSample
        set FeatureCount to featureCountSample
        set IterationCount to 125
        set LearningRate to learnRateSample
        set RegularizationTerm to regularTermSample
    }

    for(int i=0;i<100;i++) {
        algorithm("ItemItem") {
            bind ItemScorer to ItemItemScorer
            bind UserVectorNormalizer to BaselineSubtractingUserVectorNormalizer
            within (UserVectorNormalizer) {
            	bind (BaselineScorer, ItemScorer) to ItemMeanRatingItemScorer
            }
        }
    }
    algorithm("ItemMean") {
        bind ItemScorer to ItemItemScorer
            bind UserVectorNormalizer to BaselineSubtractingUserVectorNormalizer
            within (UserVectorNormalizer) {
            	bind (BaselineScorer, ItemScorer) to ItemMeanRatingItemScorer
            }
    }
}
