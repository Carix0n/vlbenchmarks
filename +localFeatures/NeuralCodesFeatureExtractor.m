classdef NeuralCodesFeatureExtractor < helpers.GenericInstaller ...
    & localFeatures.GenericLocalFeatureExtractor

  properties (SetAccess=private, GetAccess=public)
    % Set the default values of the detector options
    Opts = struct(...
      'levels', 1,... % Generated scales -- scalar or 1d array
      'netPath', '',... % Path to pre-trained model to be used
      'useGpu', false,... % Whether or not to use GPU
      'layerName', 'fc6'... % Layer which output is to be used as image descriptor
      );
    Net;
    outputSize;
  end
  
  properties (Hidden)
    layerIndex;
  end

  methods
    function obj = NeuralCodesFeatureExtractor(varargin)
      % Implement a constructor to parse any option passed to the
      % feature extractor and store them in the obj.Opts structure.

      % Information that this detector is able to extract descriptors
      obj.ExtractsDescriptors = true;
      % Name of the features extractor
      obj.Name = 'Neural codes extractor';
      % Because this class inherits methods from helpers.GenericInstalles
      % we can test whether this detector is installed
      varargin = obj.checkInstall(varargin);
      % Configure the logger. The parameters accepted by logger are
      % consumend and the rest is passed back to varargin.
      varargin = obj.configureLogger(obj.Name,varargin);
      % Parse the class options. See the properties of this class where the
      % options are defined.
      obj.Opts = vl_argparse(obj.Opts,varargin);
      % Load net
      obj.Net = load(obj.Opts.netPath);
      % Update model if necessary
      obj.Net = vl_simplenn_tidy(obj.Net);
      % Move net to GPU if specified
      if obj.Opts.useGpu, obj.Net = vl_simplenn_move(obj.Net, 'gpu'); end
      % Index of layer to be used
      obj.layerIndex = find(cellfun(@(layer) strcmp(layer.name, obj.Opts.layerName), obj.Net.layers));
      % Determine output size
      obj.outputSize = numel(obj.Net.layers{obj.layerIndex}.weights{2});
    end

    function [frames descriptors] = extractFeatures(obj, imagePath)
      import helpers.*;

      startTime = tic;
      % Because this class inherits from helpers.Logger, we can use its
      % methods for giving information to the user. Advantage of the 
      % logger is that the user can set the class verbosity. See the
      % helpers.Logger documentation.
      if nargout == 1
        obj.info('Computing frames of image %s.',getFileName(imagePath));
      else
        obj.info('Computing frames and descriptors of image %s.',...
          getFileName(imagePath));
      end
      % If you want use cache, in the first step, try to load features from
      % the cache. The third argument of loadFeatures tells whether to load
      % descriptors as well.
      [frames descriptors] = obj.loadFeatures(imagePath,nargout > 1);
      % If features loaded, we are done
      if numel(frames) > 0; return; end;
      % Get the size of the image
      imgSize = helpers.imageSize(imagePath);
      % Generate the grid of frames in all user selected grids
      for level = obj.Opts.levels
        fDist = 1 / (level + 1);
        grid = fDist:fDist:(1 - fDist);
        xGrid = imgSize(1) * grid;
        yGrid = imgSize(2) * grid;
        [yCoords xCoords] = meshgrid(xGrid,yGrid);
        detFrames = [xCoords(:)';
            yCoords(:)';
            fDist * imgSize(2) * ones(1, numel(xCoords)) - 1;
            zeros(1, numel(xCoords));
            fDist * imgSize(1) * ones(1, numel(yCoords)) - 1];
        frames = [frames detFrames];
      end

      % If the mehod is called with two output arguments, descriptors shall
      % be calculated
      if nargout > 1
        [frames descriptors] = obj.extractDescriptors(imagePath,frames);
      end
      timeElapsed = toc(startTime);
      
      obj.debug(sprintf('Features from image %s computed in %gs',...
        getFileName(imagePath),timeElapsed));
      % Store the generated frames and descriptors to the cache.
      obj.storeFeatures(imagePath, frames, descriptors);
    end

    function [frames descriptors] = extractDescriptors(obj, imagePath, frames)
      % EXTRACTDESCRIPTORS Compute mean, variance and median of the integer
      %   disk frame.
      %
      %   This is mean as an example how to work with the detected frames.
      %   The computed descriptor bears too few data to be distinctive.
      import localFeatures.helpers.*;
      obj.info('Computing descriptors.');
      startTime = tic;
      % Get the input image
      img = single(imread(imagePath));

      % Prealocate descriptors
      
      descriptors = zeros(obj.outputSize, size(frames, 2), 'single');
      
      if obj.Opts.useGpu
        descriptors = gpuArray(descriptors);
      end

      % Compute the descriptors as mean and variance of the image box
      % determined by the integer frame scale
      for fidx = 1:size(frames, 2)
        x = round(frames(1, fidx));
        y = round(frames(2, fidx));
        dx = floor(frames(3, fidx));
        dy = floor(frames(5, fidx));
        
        patch = imresize(img(y-dy:y+dy, x-dx:x+dx, :),...
            [obj.Net.meta.normalization.imageSize(1), obj.Net.meta.normalization.imageSize(2)]);
        
        averageImage = obj.Net.meta.normalization.averageImage;
      
        if ismatrix(averageImage) && ndims(averageImage) < 3
            repAverageImage = zeros(obj.Net.meta.normalization.imageSize);

            for dim = 1:obj.Net.meta.normalization.imageSize(3)
                repAverageImage(:, :, dim) = repmat(averageImage(dim), obj.Net.meta.normalization.imageSize(1:2));
            end

            averageImage = repAverageImage;
        end
        
        if obj.Opts.useGpu
            patch = gpuArray(patch);
            averageImage = gpuArray(averageImage);
        end
        
        patch = patch - averageImage;    
        nn_outputs = vl_simplenn(obj.Net, patch);
        descriptor = nn_outputs(obj.layerIndex + 1).x;
        descriptor = reshape(descriptor, [numel(descriptor), 1]);
        descriptors(:, fidx) = descriptor / norm(descriptor);
        
      end
      
      if obj.Opts.useGpu
          descriptors = gather(descriptors);    
      end
      
      elapsedTime = toc(startTime);
      obj.debug('Descriptors computed in %gs',elapsedTime);
      % This method does not cache the computed values as it is complicated
      % to compute a signature of the input frames.
    end

    function signature = getSignature(obj)
      % This method is called from loadFeatures and  storeFeatures methods
      % to ge uniqie string for the detector properties. Because this is
      % influenced both by the detector settings and its implementation,
      % the string signature of both of them.
      % fileSignature returns a string which contain information about the
      % file including the last modification date.
      signature = [helpers.struct2str(obj.Opts),';',...
        helpers.fileSignature(mfilename('fullpath'))];
    end
  end

  methods(Static)
    %  Because this class is is subclass of GenericInstaller it can benefit
    %  from its support. When GenericInstaller.install() method is called,
    %  the following operations are performed when it was detected that the
    %  class is not installed:
    %
    %    1. Install dependencies
    %    2. Download and unpack tarballs
    %    3. Run compilation
    %    4. Compile mex files
    %    5. Setup the class
    %
    %  These steps are defined by the following static methods
    %  implementations:
    %
    %   deps = getDependencies()
    %     Define the dependencies, i.e. instances of GenericInstaller which
    %     are installed when method install() is called.
    %
    %   [urls dstPaths] = getTarballsList()
    %     Returns urls = {archive_1_url, archive_2_url,...} and 
    %     dstPaths = {archive_1_dst_path,...} and defines which files
    %     should be downloaded when install() is called.
    %
    %   compile()
    %     User defined method which is called after installing all tarballs
    %     and when isCompiled returns false.
    %
    %   res = isCompiled()
    %     User defined method to test whether compile() method should be
    %     called to complete the class isntallation.
    %
    %   [srclist flags]  = getMexSources()
    %     Returns srclist = {path_to_mex_files} and their flags which are
    %     compiled using mex command. See helpers.Installer for an example.
    %
  end
end
