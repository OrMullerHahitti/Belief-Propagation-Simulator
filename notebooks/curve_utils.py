import matplotlib.pyplot as plt
import numpy as np

def get_line_segments(slopes, intercepts):
    """
    Returns the segments (m, c, start, end) that make up the lower envelope for a single row.
    """
    lines = sorted(zip(slopes, intercepts), key=lambda x: (x[0], x[1]), reverse=True)
    
    unique_lines = []
    if lines:
        unique_lines.append(lines[0])
        for i in range(1, len(lines)):
            if lines[i][0] == lines[i-1][0]: continue 
            unique_lines.append(lines[i])
    
    stack = [] 
    for m_curr, c_curr in unique_lines:
        while len(stack) > 0:
            start_prev, m_prev, c_prev = stack[-1]
            x_int = (c_curr - c_prev) / (m_prev - m_curr)
            if x_int <= start_prev:
                stack.pop()
            else:
                stack.append((x_int, m_curr, c_curr))
                break
        if not stack:
            stack.append((-np.inf, m_curr, c_curr))
            
    segments = []
    for i in range(len(stack)):
        start = stack[i][0]
        end = stack[i+1][0] if i + 1 < len(stack) else np.inf
        segments.append({'start': start, 'end': end, 'm': stack[i][1], 'c': stack[i][2]})
    return segments

def find_curve_intersections(segments_1, segments_2):
    """
    Finds exact points where two piecewise linear functions cross.
    """
    crossings = []
    
    for s1 in segments_1:
        for s2 in segments_2:
            # Check if x-intervals overlap
            overlap_start = max(s1['start'], s2['start'])
            overlap_end = min(s1['end'], s2['end'])
            
            if overlap_start < overlap_end:
                # Solve: m1*x + c1 = m2*x + c2
                # x * (m1 - m2) = c2 - c1
                dm = s1['m'] - s2['m']
                dc = s2['c'] - s1['c']
                
                if abs(dm) > 1e-9: # If lines are not parallel
                    x_cross = dc / dm
                    
                    # Check if solution is inside the overlapping interval
                    # Use a small epsilon for float comparison
                    if overlap_start - 1e-5 <= x_cross <= overlap_end + 1e-5:
                        y_cross = s1['m'] * x_cross + s1['c']
                        crossings.append((x_cross, y_cross))
                        
    return crossings

def plot_curve_intersections(cost_matrix, axis=1, edge_values=None, delta_q_range=(-15, 15)):
    M = np.array(cost_matrix)
    if axis == 0: M = M.T
    n_out, n_in = M.shape
    
    if edge_values is None: edge_values = np.arange(n_in)
    else: edge_values = np.array(edge_values)

    # 1. Get Segments for all rows
    all_row_segments = []
    for i in range(n_out):
        all_row_segments.append(get_line_segments(edge_values, M[i]))

    # 2. Find Intersections between EVERY pair of rows
    intersections = []
    for i in range(n_out):
        for j in range(i + 1, n_out):
            pts = find_curve_intersections(all_row_segments[i], all_row_segments[j])
            for pt in pts:
                intersections.append({'x': pt[0], 'y': pt[1], 'labels': (chr(97+i), chr(97+j))})

    # ================= PLOTTING =================
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_out))
    
    # Plot Curves
    x_grid = np.linspace(delta_q_range[0], delta_q_range[1], 1000)
    for i in range(n_out):
        # Reconstruct curve for plotting
        y_vals = np.zeros_like(x_grid)
        for idx, x in enumerate(x_grid):
            # Evaluate min-sum for this specific x
            y_vals[idx] = np.min(M[i, :] + edge_values * x)
            
        plt.plot(x_grid, y_vals, color=colors[i], label=f'{chr(97+i)}', linewidth=2.5)

    # Plot INTERSECTIONS (The requested highlights)
    for pt in intersections:
        x, y = pt['x'], pt['y']
        
        # Only plot if within view
        if delta_q_range[0] <= x <= delta_q_range[1]:
            # Big Red Dot
            plt.plot(x, y, 'o', color='red', markersize=10, zorder=20, alpha=0.8)
            plt.plot(x, y, 'x', color='white', markersize=6, zorder=21)
            
            # Dashed Drop Line
            plt.vlines(x, plt.ylim()[0], y, colors='red', linestyles=':', alpha=0.5)
            
            # Label
            pair_label = f"{pt['labels'][0]} vs {pt['labels'][1]}"
            plt.annotate(f"Crossing\nx={x:.2f}\n({pair_label})", 
                         xy=(x, y), 
                         xytext=(x, y + 5),
                         ha='center', fontsize=9, fontweight='bold', color='darkred',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
                         arrowprops=dict(arrowstyle="->", color='red'))

    plt.title("Highlighted Curve Crossings (Value Intersections)", fontsize=14)
    plt.xlabel(r"$\Delta Q$", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.xlim(delta_q_range)
    plt.ylim(bottom=np.min(M)-20)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Variable")
    plt.tight_layout()
    plt.show()
