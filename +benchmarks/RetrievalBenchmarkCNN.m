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
    function obj = RetrievalBenchmarkCNN(varargin)
      import helpers.*;
      obj.BenchmarkName = 'RetrBenchmarkCNN';
      [obj.Opts varargin] = vl_argparse(obj.Opts,varargin);
      varargin = obj.configureLogger(obj.BenchmarkName,varargin);
      obj.checkInstall(varargin);
    end

    function [mAP, info] = testFeatureExtractor(obj, featExtractor, dataset)
      import helpers.*;
      
      if isfield(featExtractor.Opts, 'levels')
          obj.NumPatchesPerImage = sum(featExtractor.Opts.levels .^ 2);
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
      ids = cell(numChunks,1); % as image indexes
      dists = cell(numChunks,1);
      numDescriptors = cell(1,numChunks);
      obj.info('Dataset has been divided into %d chunks.',numChunks);

      % Load query descriptors
      queryDescriptors = obj.gatherQueriesDescriptors(dataset, featExtractor);
      numQueryDescriptors = cellfun(@(a) size(a,2),queryDescriptors);

      % Compute dists for all image chunks
      for chNum = 1:numChunks
        firstImageNo = (chNum-1)*imgsPerChunk+1;
        lastImageNo = min(chNum*imgsPerChunk,numImages);
        [ids{chNum}, dists{chNum}, numDescriptors{chNum}] = ...
          obj.computeDistances(dataset,featExtractor,queryDescriptors,firstImageNo,lastImageNo);
      end
      % Compute the AP
      numDescriptors = cell2mat(numDescriptors);
      numQueries = dataset.NumQueries;
      obj.info('Computing the average precisions.');
      queriesAp = zeros(1,numQueries);
      rankedLists = zeros(numImages,numQueries);
      votes = zeros(numImages,numQueries);
      for q=1:numQueries
        % Combine distances of all descriptors from all chunks
        allQIds = cell(numChunks,1);
        allQDists = cell(numChunks,1);
        for ch = 1:numChunks
          allQIds{ch} = ids{ch}{q}';
          allQDists{ch} = dists{ch}{q}';
        end
        allQIds = cell2mat(allQIds(~cellfun('isempty',allQIds)));
        allQDists = cell2mat(allQDists(~cellfun('isempty',allQDists)));

        % Sort them by the distance to the query descriptors
        [allQDists ind] = sort(allQDists,1,'ascend');
        allQIds = allQIds(ind);

        query = dataset.getQuery(q);
        [queriesAp(q) rankedLists(:,q) votes(:,q)] = ...
          obj.computeAp(allQIds, allQDists, numDescriptors, query);
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
    function qDescriptors = gatherQueriesDescriptors(obj, dataset, featExtractor)
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
        obj.info('Query %d: %d features.',q,size(qDescriptors{q},2));
      end
    end

    function [ap rankedList votes] = computeAp(obj, imgIds, dists, numDescriptors, query)
      import helpers.*;

      numImages = numel(numDescriptors);
      
      if isempty(imgIds)
        ap = 0; rankedList = zeros(numImages,1); 
        votes = zeros(numImages,1); return;
      end
      
      votes = vl_binsum( single(zeros(numImages,1)),...
        repmat( dists(end), size(dists, 1), 1 ) - dists,...
        imgIds );
      votes = votes ./ sqrt(max(numDescriptors',1));
      [votes, rankedList]= sort(votes, 'descend'); 

      ap = obj.rankedListAp(query, rankedList);
    end
    
    function [queriesIdxs, queriesDists, numDescriptors] = ...
        computeDistances(obj, dataset, featExtractor, qDescriptors, firstImageNo, lastImageNo)
      
      import helpers.*;
      startTime = tic;
      numQueries = dataset.NumQueries;
      numImages = lastImageNo - firstImageNo + 1;
      queriesIdxs = cell(1,numQueries);
      queriesDists = cell(1,numQueries);

      testSignature = obj.getSignature;
      detSignature = featExtractor.getSignature;
      imagesSignature = dataset.getImagesSignature(firstImageNo:lastImageNo);

      % Try to load already computed queries
      isCachedQuery = false(1,numQueries);
      nonCachedQueries = 1:numQueries;
      distsResKeys = cell(1,numQueries);
      imgsInfoKey = strcat(obj.DatasetChunkInfoPrefix, testSignature, detSignature, imagesSignature);
      cacheResults = featExtractor.UseCache && obj.UseCache;
      if cacheResults
        for q = 1:numQueries
          querySignature = dataset.getQuerySignature(q);
          distsResKeys{q} = strcat(obj.QueryDistancesPrefix, testSignature,...
              detSignature, imagesSignature, querySignature);
          qDistsResults = DataCache.getData(distsResKeys{q});
          if ~isempty(qDistsResults);
            isCachedQuery(q) = true;
            [queriesIdxs{q}, queriesDists{q}] = qDistsResults{:};
            obj.debug('Query distances for images %d:%d loaded from cache.',...
              firstImageNo, lastImageNo);
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
      [descriptors, imageIdxs, numDescriptors] = ...
        obj.getDatasetFeatures(dataset,featExtractor,firstImageNo,lastImageNo);

      if cacheResults
        DataCache.storeData(imgsInfoKey,numDescriptors);
      end

      % Compute distances
      helpers.DataCache.disableAutoClear();
      queriesDistsTmp = cell(1,numel(nonCachedQueries)) ;
      queriesIdxsTmp = cell(1,numel(nonCachedQueries)) ;
      parfor qi = 1:numel(nonCachedQueries)
        q = nonCachedQueries(qi) ;
        obj.info('Imgs %d:%d - Computing distances for query %d/%d.',...
          firstImageNo,lastImageNo,q,numQueries);
        queriesDistsTmp{qi} = ...
          obj.computeDistance(descriptors, qDescriptors{q});
        queriesIdxsTmp{qi} = firstImageNo:lastImageNo;
        if cacheResults
          DataCache.storeData({queriesIdxsTmp{qi}, queriesDistsTmp{qi}},...
            distsResKeys{q});
        end
      end
      queriesDists(nonCachedQueries) = queriesDistsTmp ;
      queriesIdxs(nonCachedQueries) = queriesIdxsTmp ;
      clear queriesDistsTmp queriesIdxsTmp ;
      helpers.DataCache.enableAutoClear();
      obj.debug('All distances for %d images computed in %gs.',...
        numImages, toc(startTime));
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

      dists = obj.imageDistances(qDescriptors,descriptors,numImages);

      obj.debug('Distances calculated in %fs.',toc(startTime));
    end

    function [descriptors, imageIdxs, numDescriptors] = ...
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
  
  methods (Access = private)
      function dists = imageDistances(obj, qDescriptors, descriptors, numImages)
        descriptorsPairwiseDistMatrix = obj.pairwiseDistance(qDescriptors, descriptors);
        imagesPairwiseDistTable = reshape(descriptorsPairwiseDistMatrix, ...
            [obj.NumPatchesPerImage, obj.NumPatchesPerImage, numImages]);
        minDescToDescDistanceMatrix = reshape(min(imagesPairwiseDistTable, [], 2), ...
            [obj.NumPatchesPerImage, numImages]);
        dists = mean(minDescToDescDistanceMatrix);
      end
  end
  
  methods (Access = protected)
    function deps = getDependencies(obj)
      import helpers.*;
      deps = {Installer(),benchmarks.helpers.Installer(),...
        VlFeatInstaller('0.9.15'),YaelInstaller()};
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

      [recall, precision, info] = vl_pr(rankedLabels, numImages:-1:1);
      ap = info.auc;
    end
    
    function [auc, tpr, tnr] = rankedListRoc(query, rankedList)
      % rankedListRoc Calculate roc-score of retrieved images
      %   AUC = rankedListRoc(QUERY, RANKED_LIST) Compute area under ROC curve 
      %   of retrieved images (their ids) by QUERY, sorted by their 
      %   relevancy in RANKED_LIST.
      %
      %   [AUC TPR TNR] = rankedListAp(...) Return also true-positive and
      %   true-negative rates.

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

      [tpr, tnr, info] = vl_roc(rankedLabels, numImages:-1:1);
      auc = info.auc;
    end
  end
  
  methods (Static, Access = private)
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
end

