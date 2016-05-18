classdef RetrievalBenchmarkCNN < benchmarks.GenericBenchmark ...
    & helpers.GenericInstaller & helpers.Logger

  properties
    % Object options
    Opts = struct(...
      'maxNumImagesPerSearch',100);
  end

  properties (Constant, Hidden)
    % Key prefix for final results
    ResultsKeyPrefix = 'retreivalResults.';
    % Key prefix for distance computation results (most time consuming)
    QueryDistancesPrefix = 'retreivalDistances';
    % Key prefix for additional information about the detector features
    DatasetChunkInfoPrefix = 'datasetChunkInfo';
  end
  
  properties (Hidden)
      NumPatchesPerImage = 1;
  end

  methods
    function obj = RetrievalBenchmark(varargin)
      import helpers.*;
      obj.BenchmarkName = 'RetrBenchmarkCNN';
      [obj.Opts varargin] = vl_argparse(obj.Opts,varargin);
      varargin = obj.configureLogger(obj.BenchmarkName,varargin);
      obj.checkInstall(varargin);
    end

    function [mAP, info] = ...
        testFeatureExtractor(obj, featExtractor, dataset)
      import helpers.*;
      
      if exist('obj.Opts.levels', 'var')
          obj.NumPatchesPerImage = sum(obj.Opts.levels .^ 2);
      end
      
      obj.info('Evaluating detector %s on dataset %s.',...
        featExtractor.Name, dataset.DatasetName);
      startTime = tic;

      % Try to load results from cache
      numImages = dataset.NumImages;
      testSignature = obj.getSignature;
      detSignature = featExtractor.getSignature;
      obj.info('Computing signatures of %d images.',numImages);
      imagesSignature = dataset.getImagesSignature();
      queriesSignature = dataset.getQueriesSignature();
      resultsKey = strcat(obj.ResultsKeyPrefix, testSignature, ...
        detSignature, imagesSignature, queriesSignature);
      if obj.UseCache
        results = DataCache.getData(resultsKey);
        if ~isempty(results)
          [mAP, info]=results{:};
          obj.debug('Results loaded from cache.');
          return;
        end
      end
      % Divide the dataset into chunks
      imgsPerChunk = min(obj.Opts.maxNumImagesPerSearch,numImages);
      numChunks = ceil(numImages/imgsPerChunk);
      knns = cell(numChunks,1); % as image indexes
      knnDists = cell(numChunks,1);
      numDescriptors = cell(1,numChunks);
      obj.info('Dataset has been divided into %d chunks.',numChunks);

      % Load query descriptors
      queryDescriptors = obj.gatherQueriesDescriptors(dataset, featExtractor);
      numQueryDescriptors = cellfun(@(a) size(a,2),queryDescriptors);

      % Compute KNNs for all image chunks
      for chNum = 1:numChunks
        firstImageNo = (chNum-1)*imgsPerChunk+1;
        lastImageNo = min(chNum*imgsPerChunk,numImages);
        [knns{chNum}, knnDists{chNum}, numDescriptors{chNum}] = ...
          obj.computeKnns(dataset,featExtractor,queryDescriptors,...
          firstImageNo,lastImageNo);
      end
      % Compute the AP
      numDescriptors = cell2mat(numDescriptors);
      numQueries = dataset.NumQueries;
      obj.info('Computing the average precisions.');
      queriesAp = zeros(1,numQueries);
      rankedLists = zeros(numImages,numQueries);
      votes = zeros(numImages,numQueries);
      for q=1:numQueries
        % Combine knns of all descriptors from all chunks
        allQKnns = cell(numChunks,1);
        allQKnnDists = cell(numChunks,1);
        for ch = 1:numChunks
          allQKnns{ch} = knns{ch}{q};
          allQKnnDists{ch} = knnDists{ch}{q};
        end
        allQKnns = cell2mat(allQKnns(~cellfun('isempty',allQKnns)));
        allQKnnDists = cell2mat(allQKnnDists(~cellfun('isempty',allQKnnDists)));

        % Sort them by the distance to the query descriptors
        [allQKnnDists ind] = sort(allQKnnDists,1,'ascend');
        for qd = 1:size(allQKnnDists,2)
           allQKnns(:,qd) = allQKnns(ind(:,qd),qd);
        end
        % Pick the upper k
        fk = min(obj.Opts.k,size(allQKnns,2));
        allQKnns = allQKnns(1:fk,:);
        allQKnnDists = allQKnnDists(1:fk,:);

        query = dataset.getQuery(q);
        [queriesAp(q) rankedLists(:,q) votes(:,q)] = ...
          obj.computeAp(allQKnns, allQKnnDists, numDescriptors, query);
        obj.info('Average precision of query %d: %f',q,queriesAp(q));
      end

      mAP = mean(queriesAp);
      obj.debug('mAP computed in %fs.',toc(startTime));
      obj.info('Computed mAP is: %f',mAP);

      info = struct('queriesAp',queriesAp,'rankedList',rankedLists,...
        'votes',votes, 'numDescriptors',numDescriptors,...
        'numQueryDescriptors',numQueryDescriptors);

      results = {mAP, info};
      if obj.UseCache
        DataCache.storeData(results, resultsKey);
      end
    end

    function signature = getSignature(obj)
      signature = helpers.struct2str(obj.Opts);
    end
  end

  methods (Access=protected, Hidden)
    function qDescriptors = gatherQueriesDescriptors(obj, dataset, ...
        featExtractor)
      % gatherQueriesDescriptors Compute queries descriptors
      %   Q_DESCRIPTORS = obj.gatherQueriesDescriptors(DATASET,FEAT_EXTRACT)
      %   computes Q_DESCRIPTORS, cell array of size 
      %   [1, DATASET.NumQueries] where each cell contain descriptors from
      %   the query bounding box computed in the query image using feature
      %   extractor FEAT_EXTRACTOR.
      import benchmarks.*;
      % Gather query descriptors
      obj.info('Computing query descriptors.');
      numQueries = dataset.NumQueries;
      qDescriptors = cell(1,numQueries);
      for q=1:numQueries
        query = dataset.getQuery(q);
        bbox = query.box + 1;
        imgPath = dataset.getImagePath(query.imageId);
        [qFrames qDescriptors{q}] = featExtractor.extractFeatures(imgPath);
        % Pick only features in the query box
        visibleFrames = ...
          bbox(1) < qFrames(1,:) & ...
          bbox(2) < qFrames(2,:) & ...
          bbox(3) > qFrames(1,:) & ...
          bbox(4) > qFrames(2,:) ;
        qDescriptors{q} = qDescriptors{q}(:,visibleFrames);
        obj.info('Query %d: %d features.',q,size(qDescriptors{q},2));
      end
    end

    function [ap rankedList votes] = computeAp(obj, knnImgIds, knnDists,...
        numDescriptors, query)
      % computeAp Compute average precision from KNN results
      %   [AP RANKED_LIST VOTES] = obj.computeAp(KNN_IMG_IDS, KNN_DISTS,
      %      NUM_DESCRIPTORS, QUERY) Compute average precision of the
      %   results of K-nearest neighbours search. Result of this search is
      %   set of K descriptors for each query descriptors.
      %   Array KNN_IMG_IDS has size [K,QUERY_DESCRIPTORS_NUM] and value
      %   KNN_IMG_IDS(N,I) is the ID of the image in which the
      %   N-nearest neighbour desc. of the Ith query descriptor was found.
      %   Array KNN_DISTS has size [K,QUERY_DESCRIPTORS_NUM] and value
      %   KNN_DISTS(N,I) is the distance of the N-Nearest descriptor to the
      %   Ith query descriptor.
      %   Array NUM_DESCRIPTORS of size [1, NUM_IMAGES_IN_DB] contains the
      %   number of descriptors extracted from the database images.
      import helpers.*;

      if isempty(knnImgIds)
        numImages = numel(numDescriptors);
        ap = 0; rankedList = zeros(numImages,1); 
        votes = zeros(numImages,1); return;
      end

      k = obj.Opts.k;
      numImages = numel(numDescriptors);
      qNumDescriptors = size(knnImgIds,2);

      votes= vl_binsum( single(zeros(numImages,1)),...
        repmat( knnDists(end,:), min(k,qNumDescriptors), 1 ) - knnDists,...
        knnImgIds );
      votes = votes./sqrt(max(numDescriptors',1))./sqrt(max(qNumDescriptors,1));
      [votes, rankedList]= sort(votes, 'descend'); 

      ap = obj.rankedListAp(query, rankedList);
    end

    function [queriesKnns, queriesKnnDists numDescriptors] = ...
        computeKnns(obj, dataset, featExtractor, qDescriptors, firstImageNo,...
        lastImageNo)
      % computeKnns Compute the K-nearest neighbours of query descriptors
      %   [QUERIES_KNNS KNNS_DISTS, NUM_DESCRIPTORS] = computeKnns(DATASET,
      %     FEAT_EXTRACTOR, QUERIES_DESCRIPTORS, FIRST_IMG_NO, LAST_IMG_NO)
      %   computes KNN of all query descriptors in the database from all
      %   descriptors extracted from images [FIRST_IMG_NO, LAST_IMG_NO].
      %
      %   QUERIES_DESCRIPTORS is a cell array of size [1, DATASET.NumQueries]
      %   Array QUERIES_DESCRIPTORS{QID} contain all the descriptors 
      %   QID_DESCRIPTORS extracted by FEAT_EXTRACTOR in the query QID 
      %   bounding box. This array size is [DESC_SIZE,QID_DESCRIPTORS_NUM].
      %
      %   QUERIES_KNNS and KNNS_DISTS are cell arrays of size 
      %   [1, DATASET.NumQueries].
      %   Array QUERIES_KNNS{QID} has size [K,QID_DESCRIPTORS_NUM] and value
      %   QUERIES_KNNS{QID}(N,I) is the ID of the image in which the
      %   N-nearest neighbour desc. of the Ith query QID descriptor was found.
      %   Array KNN_DISTS has size [K,QUERY_DESCRIPTORS_NUM] and value
      %   KNN_DISTS(N,I) is the distance of the N-Nearest descriptor to the
      %   Ith query descriptor.
      %
      %   Array NUM_DESCRIPTORS of size [1, NUM_IMAGES_IN_DB] contains the
      %   number of descriptors extracted from the database images.
      import helpers.*;
      startTime = tic;
      numQueries = dataset.NumQueries;
      k = obj.Opts.k;
      numImages = lastImageNo - firstImageNo + 1;
      queriesKnns = cell(1,numQueries);
      queriesKnnDists = cell(1,numQueries);

      testSignature = obj.getSignature;
      detSignature = featExtractor.getSignature;
      imagesSignature = dataset.getImagesSignature(firstImageNo:lastImageNo);

      % Try to load already computed queries
      isCachedQuery = false(1,numQueries);
      nonCachedQueries = 1:numQueries;
      knnsResKeys = cell(1,numQueries);
      imgsInfoKey = strcat(obj.DatasetChunkInfoPrefix, testSignature,...
          detSignature, imagesSignature);
      cacheResults = featExtractor.UseCache && obj.UseCache;
      if cacheResults
        for q = 1:numQueries
          querySignature = dataset.getQuerySignature(q);
          knnsResKeys{q} = strcat(obj.QueryKnnsKeyPrefix, testSignature,...
            detSignature, imagesSignature, querySignature);
          qKnnResults = DataCache.getData(knnsResKeys{q});
          if ~isempty(qKnnResults);
            isCachedQuery(q) = true;
            [queriesKnns{q} queriesKnnDists{q}] = qKnnResults{:};
            obj.debug('Query KNNs %d for images %d:%d loaded from cache.',...
              q,firstImageNo, lastImageNo);
          end
        end
        nonCachedQueries = find(~isCachedQuery);
        % Try to avoid loading the features when all queries already 
        % computed, what need to be loaded only is the number of 
        % descriptors per image.
        numDescriptors = DataCache.getData(imgsInfoKey);
        if isempty(nonCachedQueries) && ~isempty(numDescriptors)
          return; 
        end;
      end

      % Retreive features of the images
      [descriptors imageIdxs numDescriptors] = ...
        obj.getDatasetFeatures(dataset,featExtractor,firstImageNo,...
        lastImageNo);

      if cacheResults
        DataCache.storeData(imgsInfoKey,numDescriptors);
      end

      % Compute the KNNs
      helpers.DataCache.disableAutoClear();
      queriesKnnDistsTmp = cell(1,numel(nonCachedQueries)) ;
      queriesKnnsTmp = cell(1,numel(nonCachedQueries)) ;
      parfor qi = 1:numel(nonCachedQueries)
        q = nonCachedQueries(qi) ;
        obj.info('Imgs %d:%d - Computing KNNs for query %d/%d.',...
          firstImageNo,lastImageNo,q,numQueries);
        [knnDescIds, queriesKnnDistsTmp{qi}] = ...
          obj.computeKnn(descriptors, qDescriptors{q});
        queriesKnnsTmp{qi} = imageIdxs(knnDescIds);
        if cacheResults
          DataCache.storeData({queriesKnnsTmp{qi}, queriesKnnDistsTmp{qi}},...
            knnsResKeys{q});
        end
      end
      queriesKnnDists(nonCachedQueries) = queriesKnnDistsTmp ;
      queriesKnns(nonCachedQueries) = queriesKnnsTmp ;
      clear queriesKnnDistsTmp queriesKnnsTmp ;
      helpers.DataCache.enableAutoClear();
      obj.debug('All %d-NN for %d images computed in %gs.',...
        k, numImages, toc(startTime));
    end
    
    function dists = computeDistance(obj, descriptors, qDescriptors)
    
      import helpers.*;
      import benchmarks.*;

      startTime = tic;

      numImages = size(descriptors,2) / obj.NumPatchesPerImage;
      qNumDescriptors = size(qDescriptors,2);

      if qNumDescriptors == 0
        obj.info('No descriptors detected in the query box.');
        dists = zeros(numImages,0);
        return;
      end

      obj.info('Computing distances between %d descs in db and %d descs.',...
        qNumDescriptors,size(descriptors,2));
      dists = obj.imageDistances(qDescriptors,descriptors,obj.NumPatchesPerImage,numImages);

      obj.debug('Distances calculated in %fs.',toc(startTime));
    end

    function [descriptors imageIdxs numDescriptors] = ...
        getDatasetFeatures(obj, dataset, featExtractor, firstImageNo, lastImageNo)
      % getDatasetFeatures Get all extr. features from the dataset
      %   [DESCS IMAGE_IDXS NUM_DESCS] = obj.getDatasetFeatures(DATASET,
      %   FEAT_EXTRACTOR,FIRST_IMG_NO, LAST_IMG_NO) Retrieves all
      %   extracted descriptors DESCS from images [FIRST_IMG_NO,LAST_IMG_NO]
      %   from the DATASET with FEAT_EXTRACTOR. 
      %   size(DESCS) = [DESC_SIZE,NUM_DESCRIPTORS].
      %
      %   Array IMAGE_IDXS of size(NUM_DESCRIPTORS,1) contain the id of the
      %   image in which the descriptor was calculated. The value
      %   NUM_DESCRIPTORS(1,IMAGE_ID) only gathers the number of extracted
      %   descriptor in an image.
      import helpers.*;
      numImages = lastImageNo - firstImageNo + 1;
      % Compute the features
      descriptorsStore = cell(1,numImages);
      featStartTime = tic;
      helpers.DataCache.disableAutoClear();
      parfor id = 1:numImages
        imgNo = firstImageNo + id - 1;
        obj.info('Computing features of image %d (%d/%d).',...
          imgNo,id,numImages);
        imagePath = dataset.getImagePath(imgNo);
        % Frames are ommited as score is computed from descs. only
        [frames descriptorsStore{id}] = ...
          featExtractor.extractFeatures(imagePath);
        descriptorsStore{id} = single(descriptorsStore{id});
      end
      helpers.DataCache.enableAutoClear();
      obj.debug('Features computed in %fs.',toc(featStartTime));
      % Put descriptors in a single array
      numDescriptors = cellfun(@(c) size(c,2),descriptorsStore);
      % Handle cases when no descriptors detected
      descriptorSizes = cellfun(@(c) size(c,1),descriptorsStore);
      if descriptorSizes==0
        descriptorsStore{descriptorSizes==0} =...
          single(zeros(max(descriptorSizes),0));
      end
      descriptors = cell2mat(descriptorsStore);
      imageIdxs = arrayfun(@(v,n) repmat(v,1,n),firstImageNo:lastImageNo,...
        numDescriptors,'UniformOutput',false);
      imageIdxs = [imageIdxs{:}];
    end
  end

  methods (Access = protected)
    function deps = getDependencies(obj)
      import helpers.*;
      deps = {Installer(),benchmarks.helpers.Installer(),...
        VlFeatInstaller('0.9.15'),YaelInstaller()};
    end
    
    function dists = imageDistances(qDescriptors, descriptors, numPatchesPerImage, numImages)
        descriptorsPairwiseDistMatrix = pairwiseDistance(qDescriptors, descriptors);
        imagesPairwiseDistTable = reshape(descriptorsPairwiseDistMatrix, ...
            [numPatchesPerImage, numPatchesPerImage, numImages]);
        minDescToDescDistanceMatrix = reshape(min(imagesPairwiseDistTable, [], 2), ...
            [numPatchesPerImage, numImages]);
        dists = mean(minDescToDescDistanceMatrix);
    end
    
    function res = pairwiseDistance(X, Y)
    % res = pairwiseDistance(X, Y)
    % Computes pairwise distance matrix for vector in columns of X and Y, i.e.
    % X is k-by-n matrix and Y is k-by-m matrix
    % result matrix is n-by-m
    %

        Xsize = size(X, 2);
        Ysize = size(Y, 2);

        % (x - y, x - y) = (x, x) - 2(x, y) + (y, y)
        X = X';
        X_norm = repmat(sum(X .^ 2, 2), 1, Ysize);
        Y_norm = repmat(sum(Y .^ 2), Xsize, 1);
        res = X_norm - 2 * X * Y + Y_norm;

    end
  end

  methods(Static)
    function [ap recall precision] = rankedListAp(query, rankedList)
      % rankedListAp Calculate average precision of retrieved images
      %   AP = rankedListAp(QUERY, RANKED_LIST) Compute average precision 
      %   of retrieved images (their ids) by QUERY, sorted by their 
      %   relevancy in RANKED_LIST. Average precision is calculated as 
      %   area under the precision/recall curve.
      %
      %   [AP RECALL PRECISION] = rankedListAp(...) Return also precision
      %   recall values.

      % make sure each image appears at most once in the rankedList
      [temp,inds]=unique(rankedList,'first');
      rankedList= rankedList( sort(inds) );

      numImages = numel(rankedList);
      labels = - ones(1, numImages);
      labels(query.good) = 1;
      labels(query.ok) = 1;
      labels(query.junk) = 0;
      labels(query.imageId) = 1;
      rankedLabels = labels(rankedList);

      [recall precision info] = vl_pr(rankedLabels, numImages:-1:1);
      ap = info.auc;
    end
  end  
end

