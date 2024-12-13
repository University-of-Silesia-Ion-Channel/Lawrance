import csv
import multiprocessing
import pickle
from re import X
import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
# import tensorflow.experimental.numpy as np
# import jax.numpy as np
# tf.experimental.numpy.experimental_enable_numpy_behavior()
import time as t

current = pickle.load(open('simulation_m20_D0.001.p', 'rb'))

class MinimumDescriptionLength:
  def __init__(self, x, BP=np.array([], dtype=np.int32), smallest_detectable_segment=1):
    self.x = x
    self.BP = BP
    self.smallest_detectable_segment = smallest_detectable_segment


  def scan_for_breaks_multithreaded(self, start, end, result):
    # print(start, end)
    BP = np.array([end], dtype=np.int32)
    BPlast = end
    t0 = start
    currentBP = BPlast
    while t0 < end:
      while True:
        # print(f"current segment = {t0}: {currentBP}")
        current_segment = self.x[t0:currentBP]
        # try first to fit a single break
        br = self.detect_break_mdl(current_segment, 'full')
        # print(f"br from single: {br}, {type(br)}.")
        # if no single bp we may try to fit in two bp's
        if br.size == 0 and len(current_segment) > 3*self.smallest_detectable_segment:
          br = self.detect_break_mdl(current_segment, 'full_two_break')
          # print(f"br from double: {br}, {type(br)}.")
            # %if there was a bp, put it in the list!
        if br.size != 0:
          # print("Found break")
          loc = br + t0
          # print(f"loc = {loc}")
          BP = np.append(BP, loc)
          # print(f"Break points after found break = {self.BP}")
          # set current bp to the first new breakpoint
          currentBP = loc[0]
          # print(f"loc = {loc}, currentBP = {currentBP}")
        else:
          # print(f"Number of breaks determined: {(max([len(self.BP)-1, 0]))}")
          break
      BP = np.sort(BP)
      
      # ... and move to the next segment
      t0 = currentBP + 1
      if currentBP != BPlast and currentBP != end:
        # print(f"Segment {start}:{end}", BP, currentBP, BPlast)
        # if the latest end point is the first of the ones we have so far, so the
        # new segment will not contain any of the detected segments
        BPlast = currentBP
        currentBP = BP[np.where(BP==currentBP)[0] + 1][0]
        # print(f"Updating Current BP to: {currentBP}")
    # print(BP)
    breaks = np.sort(BP) # po co tutaj sortować to nie wiem
    breaks = breaks[0:len(breaks)-1]
    # print(f"breaks before shenanigans for segment {start}:{end}: {breaks}" )
    breaks, stepvalue = self.stepstat_MDL(breaks)
    # print(f"breaks after shenanigans for segment {start}:{end}: {breaks}" )
    # print(f"stepvalue: {stepvalue}" )
    result.append(breaks)

  def scan_for_breaks(self, test):
    '''This functions scans the input vector 'x' for breakspoints using the Minimal Description Length Principle.
    Provides a piecewise constant approximation of the input vector x,
    assuming that x is described by a series of segments divided by descrete
    steps plus additive noise. A typical application is if x is an ionchannel
    recording where breakpoints indicat opening and closing events of the channel.
    Outputs:
    breaks = indices of break points in x.
    '''
    breaks = np.array([])
    if test:
      # csv file name
      filename = "/content/drive/MyDrive/breakpoints.csv"

      # initializing the titles and rows list

      rows = []

      # reading csv file
      with open(filename, 'r') as csvfile:
          # creating a csv reader object
          csvreader = csv.reader(csvfile)

          # extracting each data row one by one
          for row in csvreader:
              rows.append(row[0])
      breaks = np.array(rows, dtype=np.int32)
    else:
      if self.x.size == 0:
        N = 5000
        s = 2
        A = 3
        n = 20
        self.x = np.zeros((n,N), dtype=np.int64)
        for k in range(0, n):
          a0 = np.random.normal(0,A,1)
          self.x[k,:] = np.random.normal(a0,s,(1,N))
          self.smallest_detectable_segmentx = np.reshape(self.x.transpose(), (1, n*N))

      self.BP = np.array([len(self.x)], dtype=np.int32)
      BPlast = len(self.x)
      t0 = 0
      currentBP = BPlast

      while t0 < len(self.x):
        while True:
          print(f"current segment = {t0}: {currentBP}")
          current_segment = self.x[t0:currentBP]
          # try first to fit a single break
          br = self.detect_break_mdl(current_segment, 'full')
          # print(f"br from single: {br}, {type(br)}.")
          # if no single bp we may try to fit in two bp's
          if br.size == 0 and len(current_segment) > 3*self.smallest_detectable_segment:
            br = self.detect_break_mdl(current_segment, 'full_two_break')
            # print(f"br from double: {br}, {type(br)}.")
              # %if there was a bp, put it in the list!
          if br.size != 0:
            print("Found break")
            loc = br + t0
            print(f"loc = {loc}")
            self.BP = np.append(self.BP, loc)
            print(f"Break points after found break = {self.BP}")
            # set current bp to the first new breakpoint
            currentBP = loc[0]
            print(f"loc = {loc}, currentBP = {currentBP}")
          else:
            print(f"Number of breaks determined: {(max([len(self.BP)-1, 0]))}")
            break
        self.BP = np.sort(self.BP)
        # ... and move to the next segment
        t0 = currentBP + 1
        if currentBP != BPlast and currentBP != len(self.x):
          # if the latest end point is the first of the ones we have so far, so the
          # new segment will not contain any of the detected segments
          BPlast = currentBP
          # print(self.BP[np.where(self.BP==currentBP)[0]])
          currentBP = self.BP[np.where(self.BP==currentBP)[0] + 1][0]
          print(f"Updating Current BP to: {currentBP}")
      breaks = np.sort(self.BP) # po co tutaj sortować to nie wiem
      breaks = breaks[0:len(breaks)-1]

    breaks, stepvalue = self.stepstat_MDL(breaks, self.x)
    print(f"breaks after shenanigans: {breaks}" )
    print(f"stepvalue: {stepvalue}" )
    plt.figure(100)
    dt = 0.5
    T = np.arange(0, len(self.x))
    plt.plot(T, self.x, label='data')
    plt.plot(np.linspace(0, len(self.x), len(stepvalue)), np.sign(stepvalue)*np.round(np.abs(stepvalue)), label='step value')

    minx = 1.2 * min(self.x)
    maxx = 1.2 * max(self.x)


    for k in range(len(breaks)):
      plt.plot([breaks[k] + dt, breaks[k] + dt], [minx, maxx], 'k-')

    # plt.gca().set_fontsize(14)
    plt.title(f"scan_for_breaks_MDL' test run.")
    plt.legend(['data', 'O/C', 'breakpoint'])
    plt.xlabel('position')
    plt.ylabel('x')

    plt.show()

  def plot_results(self, breaks):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12,8))

    T = np.arange(len(self.x))

    # Generate toggling stepvalue
    toggle_value = np.round(self.x[0])
    stepvalue = np.full_like(self.x, toggle_value, dtype=np.int16)  # Start with -1
    for b in breaks:
        toggle_value *= -1  # Toggle between -1 and 1
        stepvalue[b:] = toggle_value  # Apply the new value from the breakpoint onward

    # Plot the main data
    ax.plot(T, self.x, label='Data', color='blue', linewidth=1)

    # Plot the stepvalue
    ax.plot(T, stepvalue, label='Step Value', color='orange', linestyle='--', linewidth=1)

    # Plot vertical lines for breakpoints
    for b in breaks:
        ax.axvline(x=b, color='k', linestyle='-', linewidth=1, label='Breakpoint' if b == breaks[0] else "")

    # Set titles and labels
    ax.set_title("MDL method", fontsize=14)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('X', fontsize=12)

    # Create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    # Adjust layout
    fig.tight_layout()

    plt.show()

    return fig, ax, np.array([T * 0.0001, x, stepvalue])
  
  def detect_break_mdl(self, x, search_method):
    '''
    This function looks for single break in a timeseries. A break is inserted if the
    MDL criterin accepts the division point.
    The function returns the breakpoint
    defined as the last point in the first segment. If no break is detected we
    return 0 as break point.
    '''
    match search_method:
      case "full":
        best_bp_indx = self.detect_single_breakpoint_mdl(x)
        # print(f"Using full - single breakpoint\nbest_bp_indx={best_bp_indx}")

      case "full_two_break":
        best_bp_indx = self.detect_double_breakpoint_mdl(x)
        # print(f"Using full_two_break - double breakpoint\nbest_bp_indx={best_bp_indx}")

      case _:
        best_bp_indx = np.array([], dtype=np.int32)

    test = self.test_breakpoint_mdl(x, best_bp_indx)
    # print(f"Test passed, using {best_bp_indx}" if test else "Test failed.")
    # If the proposed BP do not pass the MDL test we set split indx to 0;
    if test:
      split = best_bp_indx
    else:
      split = np.array([], dtype=np.int32)

    return split

  def detect_single_breakpoint_mdl(self, x):
    '''
    Detects the maximum likelihood point to split the time series x into two segments.
    smallest_detectable_segment is the smallest acceptable segment length.
    returns index of the first element of the second segment.
    '''
    # print("Using single breakpoint searching...")
    n = x.size # length of time series
    # print(n)
    mean_segment1 = np.mean(x[:self.smallest_detectable_segment])
    mean_segment2 = np.mean(x[self.smallest_detectable_segment:])
    # print(f"mean_segment1 = {mean_segment1}, mean_segment2 = {mean_segment2}")
    z1 = (x[:self.smallest_detectable_segment] - mean_segment1)
    L1 = np.multiply(z1, z1)
    # sum of squared differences from the mean in segment 1
    logL1 = np.sum(L1)
    # print(f"logL1 = {logL1}")

    z2 = (x[self.smallest_detectable_segment:] - mean_segment2)
    # print(f"z2 = {z2}")
    L2 = np.multiply(z2, z2)
    # print(f"L2 = {L2}")
    # sum of squared differences from the mean in segment 2
    logL2 = np.sum(L2)
    # print(f"logL2 = {logL2}")

    bestlog = logL1 + logL2
    # print(f"Bestlog = {bestlog}")
    best_bp_indx = np.array([], dtype=np.int32)

    for i in range(self.smallest_detectable_segment + 1, n-self.smallest_detectable_segment):
      # print(i)
      new_mean_segment1 = 1/i*((i-1)*mean_segment1 + x[i])

      diff_mean_segment1 = new_mean_segment1 - mean_segment1

      new_mean_segment2 = 1/(n-i)*((n-i+1)*mean_segment2 - x[i])

      dmean_segment2 = new_mean_segment2 - mean_segment2

      newlogL1 = logL1 + (x[i] - mean_segment1)**2 - i*diff_mean_segment1**2
      newlogL2 = logL2 - (x[i] - new_mean_segment2)**2 + (n-i+1)*dmean_segment2**2

      Nloglik = newlogL1 + newlogL2

      if Nloglik < bestlog:
        bestlog = Nloglik
        best_bp_indx = np.array([i], dtype=np.int32)
        # print(f"Single Breakpoint: best_bp_indx = {best_bp_indx}")

      mean_segment1 = new_mean_segment1
      mean_segment2 = new_mean_segment2

      logL1 = newlogL1
      logL2 = newlogL2
    # print("For ended")
    return best_bp_indx

  def detect_double_breakpoint_mdl(self, x):
    '''
    Detects the maximum likelihood point to split the time series x into two segments where s
    being the standard deviation data within the segments. smallest_detectable_segment is the smallest
    acceptanel segment length.
    returns index of the first element of the second segment and the
    likelihood of the observations with two different mean values.
    This function will run in test mode if called with no arguments
    '''
    # print("Checking for double breakpoint")
    n = x.size
    cumx = np.cumsum(x)
    cumz = np.cumsum(x**2)
    cumz_end = cumz[-1]
    cumx_end = cumx[-1]

    bestlog = np.inf
    best_bp_indx = np.array([], dtype=np.int32)

    for i in range(self.smallest_detectable_segment, n - 2 * self.smallest_detectable_segment):
        cumx_i = cumx[i]
        cumz_i = cumz[i]
        mu1 = cumx_i / (i + 1)

        logL1 = cumz_i - 2 * mu1 * cumx_i + (i + 1) * mu1**2

        for j in range(i + self.smallest_detectable_segment, n - self.smallest_detectable_segment):

            l2 = j - i
            l3 = n - j

            cumx_j = cumx[j]
            cumz_j = cumz[j]

            mu2 = (cumx_j - cumx_i) / l2
            logL2 = cumz_j - cumz_i - 2 * mu2 * (cumx_j - cumx_i) + l2 * mu2**2

            mu3 = (cumx_end - cumx_j) / l3
            logL3 = cumz_end - cumz_j - 2 * mu3 * (cumx_end - cumx_j) + l3 * mu3**2

            # Total likelihood
            Nloglik = logL1 + logL2 + logL3

            if Nloglik < bestlog:
                bestlog = Nloglik
                best_bp_indx = np.array([i, j],dtype=np.int32)
    # print(f"Final best_bp_indx = {best_bp_indx}")
    return best_bp_indx

  def test_breakpoint_mdl(self, x, best_bp_indx):
    # Gives acceptance variable A. If model is better than null-hyp, then A = 1;
    # otherwize A = 0
    if best_bp_indx.size == 0:
      return 0
    else:
      mdl1 = self.model_penalty_mdl_general(x, np.array([], dtype=np.int32))
      mdl2 = self.model_penalty_mdl_general(x, best_bp_indx)
      # print(f"mdl1 = {mdl1}, mdl2 = {mdl2}")
      # The quest is to get the lowest description length
      return mdl1 > mdl2

  def model_penalty_mdl_general(self, x, BP):
    # print(f"Checking penalty for {BP}")
    N = x.size
    BPi = np.unique(np.concatenate(([0], BP, [N]), dtype=np.int32))
    # BPi = np.unique([0] + BP + [N])
    # print(f"BPi = {BPi}")
    p = np.size(BPi) - 1
    RSS = 0
    CL = 0
    for k in range(1, len(BPi)):
      seg = [_ for _ in range(BPi[k-1], BPi[k]-1)]
      mu = np.mean(x[seg])

      RSS = RSS + np.sum( np.square(x[seg] - mu) )

      Nseg = len(seg)

      CL = CL + np.log(Nseg)
    return p*np.log(N) + 0.5*CL + N/2*np.log(RSS/N)

  def stepstat_MDL(self, BP):
    # Append the last index of x to BP as a fake breakpoint
    BP = np.append(BP, len(x))
    stepvalue = np.zeros(len(BP))

    skip = 1
    i0 = BP[0]

    for k in range(len(BP)):
        indx = np.arange(i0, BP[k])
        mindx = np.arange(i0 + skip, BP[k] - skip)
        # print(f"mindx: {i0 + skip} : {BP[k] - skip}")
        if mindx.size == 0:
            # print(f'skip ignored {indx}')
            mindx = BP[k]

        stepvalue[k] = np.mean(self.x[mindx])
        # print(f"Stepvalue of {BP[k]} is {stepvalue[k]}")
        i0 = BP[k]

    jumps = np.diff(stepvalue)
    # print(f"Jumps {jumps}")
    BP = BP[:len(BP)-1]
    return BP[np.abs(jumps) > 0.65], stepvalue


if __name__ == '__main__':
  start_time = t.time()
  x = np.array(current['x'][:200000])
  n = len(x)
  modelMDL = MinimumDescriptionLength(x, smallest_detectable_segment=300)
  core_count = 8
  manager = multiprocessing.Manager()
  result = manager.list()  # Use a managed list for variable length
  segment = n // core_count
  processes = []

  for i in range(core_count):
      start = i * segment
      if i == core_count - 1:
          end = n  # Ensure the last segment goes up to the end
      else:
          end = start + segment
      # Creating a process for each segment
      p = multiprocessing.Process(target=modelMDL.scan_for_breaks_multithreaded, args=(start, end, result))
      processes.append(p)
      p.start()

  for p in processes:
      p.join()

  final_result = np.sort(np.concatenate(result))
  # print(final_result)
  # final_result, stepvalue = modelMDL.stepstat_MDL(result)
  print(f"It took {t.time()-start_time} seconds long.")
  np.savetxt("output.csv", final_result, delimiter=",", fmt='%d')  # Use '%f' for floats
  # modelMDL.scan_for_breaks(False)
  fig, ax, data = modelMDL.plot_results(final_result)
  print(data.shape)
  np.savetxt(data.transpose, delimeter=',', fname="balls.csv")

exit()