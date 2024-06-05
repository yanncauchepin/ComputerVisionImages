import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


class KeypointsMatching():
    
    
    def __init__(self):
        pass


    '''Brute-force matching
    A brute-force matcher is a descriptor matcher that compares two sets of keypoint 
    descriptors and generates a result that is a list of matches. It is called brute-force 
    because little optimization is involved in the algorithm. For each keypoint 
    descriptor in the first set, the matcher makes comparisons to every keypoint 
    descriptor in the second set. Each comparison produces a distance value and the 
    best match can be chosen on the basis of least distance.
    More generally, in computing, the term brute-force is associated with an approach 
    that prioritizes the exhaustion of all possible combinations (for example, all the 
    possible combinations of characters to crack a password of a known length). 
    Conversely, an algorithm that prioritizes speed might skip some possibilities and 
    try to take a shortcut to the solution that seems the most plausible.
    OpenCV provides a cv2.BFMatcher class that supports several approaches to 
    brute-force feature matching.
    '''

    '''matching a logo in two images'''
    @staticmethod
    def orb_matching(image_1, image_2, show=True):
        orb = cv2.ORB_create()
        image_1_kp, image_1_descriptor = orb.detectAndCompute(image_1, None)
        image_2_kp, image_2_descriptor = orb.detectAndCompute(image_2, None)
        # Perform brute-force matching.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(image_1_descriptor, image_2_descriptor)
        # Sort the matches by distance.
        matches = sorted(matches, key=lambda x:x.distance)
        # Draw the best 25 matches.
        image_matches = cv2.drawMatches(
            image_1, image_1_kp, image_2, image_2_kp, matches[:25], image_2,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # Show the matches.
        if show:
            plt.imshow(image_matches)
            plt.show()
        return image_matches
        

    '''Now, let's consider the implementation of a modified brute-force matching algorithm 
    that adaptively chooses a distance threshold in the manner we have described. In 
    the previous section's code sample, we used the match method of the cv2.BFMatcher 
    class in order to get a list containing the single best (least-distance) match for 
    each query keypoint. By doing so, we discarded information about the distance 
    scores of all the worse possible matches – the kind of information we need for 
    our adaptive approach. Fortunately, cv2.BFMatcher also provides a knnMatch method, 
    which accepts an argument, k, that specifies the maximum number of best (least-distance) 
    matches that we want to retain for each query keypoint. (In some cases, we may 
    get fewer matches than the maximum.) KNN stands for k-nearest neighbors.
    We will use the knnMatch method to request a list of the two best matches for each 
    query keypoint. Based on our assumption that each query keypoint has, at most, one 
    correct match, we are confident that the second-best match is wrong. We multiply 
    the second-best match's distance score by a value less than 1 in order to obtain 
    the threshold.
    Then, we accept the best match as a good match only if its distant score is less 
    than the threshold. This approach is known as the ratio test.
    "The probability that a match is correct can be determined by taking the ratio 
    of the distance from the closest neighbor to the distance of the second closest." 
    '''

    
    @staticmethod
    def orb_knn_matching(image_1, image_2, show=True):
        orb = cv2.ORB_create()
        image_1_kp, image_1_descriptor = orb.detectAndCompute(image_1, None)
        image_2_kp, image_2_descriptor = orb.detectAndCompute(image_2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs_of_matches = bf.knnMatch(image_1_descriptor, image_2_descriptor, k=2)
        '''knnMatch returns a list of lists; each inner list contains at least one 
        match and no more than k matches, sorted from best (least distance) to worst. 
        The following line of code sorts the outer list based on the distance score 
        of the best matches'''
        pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)
        '''Draw the top 25 best matches, along with any second-best matches that knnMatch 
        may have paired with them. We can't use the cv2.drawMatches function because 
        it only accepts a one-dimensional list of matches; instead, we must use 
        cv2.drawMatchesKnn.'''
        image_pairs_of_matches = cv2.drawMatchesKnn(
            image_1, image_1_kp, image_2, image_2_kp, pairs_of_matches[:25], image_2,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        if show:
            plt.imshow(image_pairs_of_matches)
            plt.show()
        return image_pairs_of_matches
        '''We have not filtered out any bad matches – and, indeed, we have deliberately 
        included the second-best matches, which we believe to be bad.'''

    
    @staticmethod
    def orb_knn_ratio_test_matching(image_1, image_2, show=True):
        '''Set the threshold at 0.8 times the distance score of the second-best match. 
        If knnMatch has failed to provide a second-best match, we reject the best match 
        anyway because we are unable to apply the test.'''
        orb = cv2.ORB_create()
        image_1_kp, image_1_descriptor = orb.detectAndCompute(image_1, None)
        image_2_kp, image_2_descriptor = orb.detectAndCompute(image_2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs_of_matches = bf.knnMatch(image_1_descriptor, image_2_descriptor, k=2)
        pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)
        # Apply the ratio test.
        matches = [x[0] for x in pairs_of_matches if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]
        image_matches = cv2.drawMatches(
            image_1, image_1_kp, image_2, image_2_kp, matches[:25], image_2,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        if show:
            plt.imshow(image_matches)
            plt.show()
        return image_matches


    '''FLANN stands for Fast Library for Approximate Nearest Neighbors. "FLANN is a 
    library for performing fast approximate nearest neighbor searches in high dimensional 
    spaces. It contains a collection of algorithms we found to work best for the nearest 
    neighbor search and a system for automatically choosing the best algorithm and 
    optimum parameters depending on the dataset. FLANN is written in C++ and contains 
    bindings for the following languages: C, MATLAB,Python, and Ruby.". Although FLANN 
    is also available as a standalone library, we will use it as part of OpenCV because 
    OpenCV provides a handy wrapper for it.'''
    @staticmethod
    def flann_matching(image_1, image_2, show=True):
        sift = cv2.SIFT_create()
        image_1_kp, image_1_descriptor = sift.detectAndCompute(image_1, None)
        image_2_kp, image_2_descriptor = sift.detectAndCompute(image_2, None)
        # Define FLANN-based matching parameters.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        # Perform FLANN-based matching.
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(image_1_descriptor, image_2_descriptor, k=2)
        '''Here, we can see that the FLANN matcher takes two parameters: an indexParams 
        object and a searchParams object. These parameters, passed in the form of 
        dictionaries in Python (and structs in C++), determine the behavior of the 
        index and search objects that are used internally by FLANN to compute the matches. 
        We have chosen parameters that offer a reasonable balance between accuracy and 
        processing speed. Specifically, we are using a kernel density tree (kd-tree) 
        indexing algorithm with five trees, which FLANN can process in parallel. 
        (The FLANN documentation recommends between one tree, which would offer no 
        parallelism, and 16 trees, which would offer a high degree of parallelism if 
        the system could exploit it.)
        We are performing 50 checks or traversals of each tree. A greater number of 
        checks can provide greater accuracy but at a greater computational cost.'''
        # Prepare an empty mask to draw good matches.
        mask_matches = [[0, 0] for i in range(len(matches))]
        # Populate the mask based on David G. Lowe's ratio test.
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                mask_matches[i]=[1, 0]
        '''After performing FLANN-based matching, we apply Lowe's ratio test with a 
        multiplier of 0.7. To demonstrate a different coding style, we will use the 
        result of the ratio test in a slightly different way compared to how we did 
        in the previous section's code sample. Previously, we assembled a new list 
        with just the good matches in it. This time, we will assemble a list called 
        mask_matches, in which each element is a sublist of length k (the same k 
        that we passed to knnMatch). If a match is good, we set the corresponding 
        element of the sublist to 1; otherwise, we set it to 0.
        For example, if we have mask_matches = [[0, 0], [1, 0]], this means that we 
        have two matched keypoints; for the first keypoint, the best and second-best 
        matches are both bad, while for the second keypoint, the best match is good 
        but the second-best match is bad. Remember, we assume that all the second-best 
        matches are bad.'''
        '''It is time to draw and show the good matches. We can pass our mask_matches 
        list to cv2.drawMatchesKnn as an optional argument, as highlighted.'''
        # Draw the matches that passed the ratio test.
        image_matches = cv2.drawMatchesKnn(
            image_1, image_1_kp, image_2, image_2_kp, matches, None,
            matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
            matchesMask=mask_matches, flags=0)
        '''cv2.drawMatchesKnn only draws the matches that we marked as good (with a 
        value of 1) in our mask. Let's unveil the result.'''
        if show:
            # Show the matches.
            plt.imshow(image_matches)
            plt.show()
        return image_matches


    '''Homography: "A relation between two figures, such that to any point of the one 
    corresponds one and but one point in the other, and vice versa. Thus, a tangent 
    line rolling on a circle cuts two fixed tangents of the circle in two sets of 
    points that are homographic.". A bit clearer: homography is a condition in which 
    two figures find each other when one is a perspective distortion of the other.'''
    @staticmethod
    def flann_homography_matching(image_1, image_2, show=True):
        MIN_NUM_GOOD_MATCHES = 10
        
        # Perform SIFT feature detection and description.
        sift = cv2.SIFT_create()
        image_1_kp, image_1_descriptor = sift.detectAndCompute(image_1, None)
        image_2_kp, image_2_descriptor = sift.detectAndCompute(image_2, None)
        
        # Define FLANN-based matching parameters.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # Perform FLANN-based matching.
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(image_1_descriptor, image_2_descriptor, k=2)
        
        # Find all the good matches as per Lowe's ratio test.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) >= MIN_NUM_GOOD_MATCHES:
            src_pts = np.float32(
                [image_1_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [image_2_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            mask_matches = mask.ravel().tolist()
        
            h, w = image_1.shape
            src_corners = np.float32(
                [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(src_corners, M)
            dst_corners = dst_corners.astype(np.int32)
        
            # Draw the bounds of the matched region based on the homography.
            num_corners = len(dst_corners)
            for i in range(num_corners):
                x0, y0 = dst_corners[i][0]
                if i == num_corners - 1:
                    next_i = 0
                else:
                    next_i = i + 1
                x1, y1 = dst_corners[next_i][0]
                cv2.line(image_2, (x0, y0), (x1, y1), 255, 3, cv2.LINE_AA)
        
            # Draw the matches that passed the ratio test.
            image_matches = cv2.drawMatches(
                image_1, image_1_kp, image_2, image_2_kp, good_matches, None,
                matchColor=(0, 255, 0), singlePointColor=None,
                matchesMask=mask_matches, flags=2)
            
            if show:
                # Show the homography and good matches.
                plt.imshow(image_matches)
                plt.show()
            return image_matches
        else:
            print("Not enough matches good were found - %d/%d" % \
                  (len(good_matches), MIN_NUM_GOOD_MATCHES))
            return None
        
        
    @staticmethod
    def create_descriptors(folder):
        feature_detector = cv2.SIFT_create()
        files = []
        for (dirpath, dirnames, filenames) in os.walk(folder):
            files.extend(filenames)
        for f in files:
            KeypointsMatching.create_descriptor(folder, f, feature_detector)
    
    
    @staticmethod
    def create_descriptor(folder, image_path, feature_detector):
        if not image_path.endswith('png'):
            print('skipping %s' % image_path)
            return
        print('reading %s' % image_path)
        img = cv2.imread(os.path.join(folder, image_path),
                         cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = feature_detector.detectAndCompute(
            img, None)
        descriptor_file = image_path.replace('png', 'npy')
        np.save(os.path.join(folder, descriptor_file), descriptors)


    @staticmethod
    def scan_for_matches(folder, query):
        # create files, images, descriptors globals
        files = []
        images = []
        descriptors = []
        for (dirpath, dirnames, filenames) in os.walk(folder):
            files.extend(filenames)
            for f in files:
                if f.endswith('npy') and f != 'query.npy':
                    descriptors.append(f)
        print(descriptors)
        
        # Create the SIFT detector.
        sift = cv2.SIFT_create()
        
        # Perform SIFT feature detection and description on the
        # query image.
        query_kp, query_ds = sift.detectAndCompute(query, None)
        
        # Define FLANN-based matching parameters.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        # Create the FLANN matcher.
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Define the minimum number of good matches for a suspect.
        MIN_NUM_GOOD_MATCHES = 10
        
        greatest_num_good_matches = 0
        prime_suspect = None
        
        print('>> Initiating picture scan...')
        '''Note the use of the np.load method, which loads a specified npy file into 
        a NumPy array.'''
        for d in descriptors:
            print('--------- analyzing %s for matches ------------' % d)
            matches = flann.knnMatch(
                query_ds, np.load(os.path.join(folder, d)), k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            num_good_matches = len(good_matches)
            name = d.replace('.npy', '').upper()
            if num_good_matches >= MIN_NUM_GOOD_MATCHES:
                print('%s is a suspect! (%d matches)' % \
                    (name, num_good_matches))
                if num_good_matches > greatest_num_good_matches:
                    greatest_num_good_matches = num_good_matches
                    prime_suspect = name
            else:
                print('%s is NOT a suspect. (%d matches)' % \
                    (name, num_good_matches))
        
        if prime_suspect is not None:
            print('Prime suspect is %s.' % prime_suspect)
        else:
            print('There is no suspect.')
            
            
            
            
            
            

if __name__ == '__main__':
    pass
    
    # image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/chessboard.png'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detecting_harris_corner(image_gray)
    
    # image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/Thoune3.jpg'
    # image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detecting_surf_keypoints_descriptors(image_gray)
    
    # feature_image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/nasa_logo.png'
    # feature_image = cv2.imread(feature_image_path, cv2.IMREAD_GRAYSCALE)
    # image_path = '/home/yanncauchepin/Git/PublicProjects/ComputerVisionImages/kennedy_space_center.jpg'
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # image_knn_ratio_test_matching(feature_image, image)

    # folder = 'tattoos/'
    # create_descriptors(folder)
    
    # query = cv2.imread(os.path.join(folder, 'query.png'),
    #                cv2.IMREAD_GRAYSCALE)
    # scan_for_matches(folder, query)
    