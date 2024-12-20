import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import csv
from numba import njit

@njit
def detect_single_breakpoint(x, min_seg):
    """Detects a single breakpoint in a time series.

    Args:
        x (np.ndarray): One-dimensional input data array
        min_seg (int): Minimum segment length between breakpoints

    Returns:
        np.ndarray: Array containing the index of detected breakpoint or empty array if none found
    """
    n = x.size
    if n < 2 * min_seg:
        return np.empty(0, dtype=np.int32)

    mean1 = np.mean(x[:min_seg])
    mean2 = np.mean(x[min_seg:])
    logL1 = np.sum((x[:min_seg] - mean1)**2)
    logL2 = np.sum((x[min_seg:] - mean2)**2)
    bestlog = logL1 + logL2
    best_idx = -1

    for i in range(min_seg + 1, n - min_seg):
        new_mean1 = ((i - 1)*mean1 + x[i]) / i
        diff1 = new_mean1 - mean1

        new_mean2 = ((n - i + 1)*mean2 - x[i])/(n - i)
        diff2 = new_mean2 - mean2

        newL1 = logL1 + (x[i] - mean1)**2 - i*(diff1**2)
        newL2 = logL2 - (x[i] - new_mean2)**2 + (n - i + 1)*(diff2**2)
        Nloglik = newL1 + newL2

        if Nloglik < bestlog:
            bestlog = Nloglik
            best_idx = i

        mean1 = new_mean1
        mean2 = new_mean2
        logL1 = newL1
        logL2 = newL2

    if best_idx == -1:
        return np.empty(0, dtype=np.int32)
    out = np.empty(1, dtype=np.int32)
    out[0] = best_idx
    return out

@njit
def detect_double_breakpoint(x, min_seg):
    """Detects two breakpoints in a time series.

    Args:
        x (np.ndarray): One-dimensional input data array
        min_seg (int): Minimum segment length between breakpoints

    Returns:
        np.ndarray: Array containing indices of detected breakpoints or empty array if none found
    """
    n = x.size
    if n < 3 * min_seg:
        return np.empty(0, dtype=np.int32)

    cumx = np.cumsum(x)
    cumz = np.cumsum(x*x)
    cumx_end = cumx[-1]
    cumz_end = cumz[-1]

    bestlog = np.inf
    best_i = -1
    best_j = -1

    for i in range(min_seg, n - 2 * min_seg):
        cumx_i = cumx[i]
        cumz_i = cumz[i]
        mu1 = cumx_i/(i+1)
        logL1 = cumz_i - 2*mu1*cumx_i + (i+1)*mu1**2

        for j in range(i + min_seg, n - min_seg):
            l2 = j - i
            l3 = n - j
            cumx_j = cumx[j]
            cumz_j = cumz[j]

            mu2 = (cumx_j - cumx_i)/l2
            logL2 = cumz_j - cumz_i - 2*mu2*(cumx_j - cumx_i) + l2*mu2**2

            mu3 = (cumx_end - cumx_j)/l3
            logL3 = cumz_end - cumz_j - 2*mu3*(cumx_end - cumx_j) + l3*mu3**2

            Nloglik = logL1 + logL2 + logL3
            if Nloglik < bestlog:
                bestlog = Nloglik
                best_i = i
                best_j = j

    if best_i == -1 or best_j == -1:
        return np.empty(0, dtype=np.int32)
    out = np.empty(2, dtype=np.int32)
    out[0] = best_i
    out[1] = best_j
    return out


class MDLBreakDetector:
    """Class detecting breakpoints in a time series using the MDL method.

    Attributes:
        x (np.ndarray): One-dimensional input data array
        min_seg (int): Minimum segment length between breakpoints

    Methods:
        detect_breaks_mdl(segment, method): Detects breakpoints in a segment using MDL method
        _test_breakpoint(segment, candidate): Tests if a breakpoint is valid
        _mdl(segment, BP): Calculates MDL value for segment and breakpoints
        stepstat_mdl(BP, threshold): Calculates statistics for breakpoints
        plot_results(breaks): Creates a plot with breakpoints
        save_to_csv(bp_file, step_file, final_breaks): Saves breakpoints and stats to CSV files
    """
    def __init__(self, x, min_seg=300):
        self.x = x
        self.min_seg = min_seg

    def detect_breaks_mdl(self, segment, method):
        """Detects breakpoints in a segment using the MDL method.

        Args:
            segment (np.ndarray): One-dimensional input data array
            method (str): Method for detecting breakpoints ('full' or 'full_two_break')

        Returns:
            np.ndarray: Array containing indices of detected breakpoints or empty array if none found    
        """
        if method == "full":
            candidate = detect_single_breakpoint(segment, self.min_seg)
        elif method == "full_two_break":
            candidate = detect_double_breakpoint(segment, self.min_seg)
        else:
            return np.empty(0, dtype=np.int32)

        return candidate if self._test_breakpoint(segment, candidate) else np.empty(0, dtype=np.int32)

    def _test_breakpoint(self, segment, candidate):
        """Tests if a breakpoint is valid.

        Args:
            segment (np.ndarray): One-dimensional input data array
            candidate (np.ndarray): Array containing indices of detected breakpoints

        Returns:
            bool: True if breakpoint is valid, False otherwise
        """
        if candidate.size == 0:
            return False
        mdl_no = self._mdl(segment, np.empty(0, dtype=np.int32))
        mdl_yes = self._mdl(segment, candidate)
        return mdl_no > mdl_yes

    def _mdl(self, segment, BP):
        """Calculates MDL value for segment and breakpoints.
        
        Args:
            segment (np.ndarray): One-dimensional input data array
            BP (np.ndarray): Array containing indices of breakpoints
            
        Returns:
            float: MDL value for segment and breakpoints
        """
        N = segment.size
        BPi = np.unique(np.concatenate(([0], BP, [N])))
        p = BPi.size - 1
        RSS = 0.0
        CL = 0.0
        for k in range(1, len(BPi)):
            seg = np.arange(BPi[k-1], BPi[k]-1)
            if seg.size == 0:
                continue
            mu = np.mean(segment[seg])
            RSS += np.sum((segment[seg] - mu)**2)
            Nseg = len(seg)
            if Nseg > 0:
                CL += np.log(Nseg)

        if RSS <= 0:
            return np.inf
        return p*np.log(N) + 0.5*CL + (N/2)*np.log(RSS/N)

    def stepstat_mdl(self, BP, threshold=0.8):
        """Calculates statistics for breakpoints.
        
        Args:
            BP (np.ndarray): Array containing indices of breakpoints
            threshold (float): Threshold for filtering breakpoints
            
        Returns:
            np.ndarray: Array containing filtered breakpoints
            np.ndarray: Array containing step values
        """
        BP = np.append(BP, len(self.x))
        stepvalue = np.zeros(len(BP))
        skip = 1
        i0 = BP[0]
        for k in range(len(BP)):
            start = i0 + skip
            stop = BP[k] - skip
            if stop < start:
                start = BP[k]
                stop = BP[k]
            indices = np.arange(start, stop+1)
            if indices.size == 0:
                indices = np.array([BP[k]], dtype=int)
            stepvalue[k] = np.mean(self.x[indices])
            i0 = BP[k]

        jumps = np.diff(stepvalue)
        filtered = BP[:-1][np.abs(jumps) > threshold]
        return filtered, stepvalue

    def plot_results(self, breaks):
        """Creates a plot with breakpoints.
        
        Args:
            breaks (np.ndarray): Array containing indices of breakpoints
        
        Returns:
            None
        """
        T = np.arange(len(self.x))
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(T, self.x, label='Data', color='blue')

        toggle_value = round(self.x[0])
        stepvalue = np.full_like(self.x, toggle_value, dtype=float)
        for b in breaks:
            toggle_value *= -1
            stepvalue[b:] = toggle_value

        ax.plot(T, stepvalue, label='Step Value', color='orange', linestyle='--')
        for i, b in enumerate(breaks):
            label = 'Breakpoint' if i == 0 else ""
            ax.axvline(x=b, color='k', linestyle='-', linewidth=1, label=label)

        ax.set_title("MDL method")
        ax.set_xlabel('Position')
        ax.set_ylabel('X')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        plt.show()

    def save_to_csv(self, bp_file, step_file, final_breaks):
        """Saves breakpoints and stats to CSV files.

        Args:
            bp_file (str): File name for breakpoints
            step_file (str): File name for step values
            final_breaks (np.ndarray): Array containing indices of breakpoints

        Returns:
            None

        TODO: 
            * Add header to CSV files
            * Prefferably make it make it one file with two columns 
        """
        with open(bp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(final_breaks)

        _, stepvalue = self.stepstat_mdl(final_breaks)
        with open(step_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(stepvalue)


def process_segment(args):
    """Processes a segment of the time series.
    
    Args:
        args (tuple): Tuple containing time series segment and start and end indices
        
    Returns:
        np.ndarray: Array containing indices of detected breakpoints
    """
    x, start, end, min_seg = args
    detector = MDLBreakDetector(x, min_seg=min_seg)
    BP_local = np.array([end], dtype=np.int32)
    BPlast = end
    t0 = start
    currentBP = BPlast

    while t0 < end:
        while True:
            current_segment = x[t0:currentBP]
            br = detector.detect_breaks_mdl(current_segment, 'full')
            if br.size == 0 and len(current_segment) > 3 * min_seg:
                br = detector.detect_breaks_mdl(current_segment, 'full_two_break')

            if br.size != 0:
                loc = br + t0
                BP_local = np.append(BP_local, loc)
                currentBP = loc[0]
            else:
                break

        BP_local = np.sort(BP_local)
        t0 = currentBP + 1
        if currentBP != BPlast and currentBP != end:
            BPlast = currentBP
            idx = np.where(BP_local == currentBP)[0]
            if idx.size > 0 and idx[0] + 1 < len(BP_local):
                currentBP = BP_local[idx[0] + 1]

    return np.sort(BP_local)[:-1]


if __name__ == "__main__":
    with open('simulation_m20_D0.001.p', 'rb') as f:
        current = pickle.load(f)

    x = np.array(current['x'])
    start_time = time.time()
    core_count = 8
    n = len(x)
    segment = n // core_count
    min_seg = 300

    args_list = []
    for i in range(core_count):
        start = i * segment
        end = n if i == core_count - 1 else start + segment
        args_list.append((x, start, end, min_seg))

    with multiprocessing.Pool(core_count) as pool:
        results = pool.map(process_segment, args_list)

    if len(results) > 0:
        combined = np.sort(np.concatenate(results))
        detector = MDLBreakDetector(x, min_seg=min_seg)
        final_breaks, stepvalue = detector.stepstat_mdl(combined)

        detector.save_to_csv("breakpoints.csv","stepvalues.csv", final_breaks)

        elapsed = time.time() - start_time
        print(f"Finished in {elapsed:.2f} seconds.")

        detector.plot_results(final_breaks)
    else:
        print("No breaks found.")
