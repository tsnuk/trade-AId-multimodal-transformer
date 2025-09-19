"""experimental_utils.py

Experimental utility functions that are preserved for potential future use.

These functions are kept commented out because their inputs and interfaces
do not match the current code structure, but they contain potentially useful
logic that may be adapted for future features.

"""

# def kmeans_cluster_with_sorted_centers(data, n_clusters, outlier_threshold_percentile=0.1):
#   """
#   Apply KMeans clustering to the input data and sort clusters based on the
#   minimum values within each cluster.
#
#   Args:
#     data: A list of numeric data points.
#     n_clusters: The number of clusters to create.
#     outlier_threshold_percentile: The percentile threshold for capping outliers (default: 0.1).
#
#   Returns:
#     sorted_labels: A list of cluster labels sorted based on the cluster centers.
#     sorted_clusters: A list of clusters sorted based on their center values.
#   """
#   capped_data = cap_outliers(data, outlier_threshold_percentile)
#
#   # Convert to numpy array and reshape for KMeans
#   reshaped_data = np.array(capped_data).reshape(-1, 1)
#
#   # Apply KMeans clustering
#   kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#   kmeans.fit(reshaped_data)
#
#   # Get cluster labels and centers
#   labels = kmeans.labels_
#   centers = kmeans.cluster_centers_
#
#   # Sort cluster centers and get sorted indices
#   sorted_indices = np.argsort(centers.flatten())
#
#   # Create a mapping from original labels to sorted labels
#   label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
#
#   # Reassign labels based on the sorted centers
#   sorted_labels = np.array([label_mapping[label] for label in labels])
#
#   # Create sorted clusters based on new labels
#   sorted_clusters = [[] for _ in range(n_clusters)]
#   for i, label in enumerate(sorted_labels):
#       # Ensure index is within bounds of the original data list
#       if i < len(capped_data):
#           sorted_clusters[label].append(capped_data[i])
#       else:
#           # This case should not happen if sorted_labels has the same length as capped_data
#           print(f"Warning: Index {i} out of bounds for data list in sorted_clusters creation.")
#
#
#   # Print cluster information
#   print('\n--- KMeans Clustering Results ---')
#   sorted_cluster_ranges = []
#   sorted_cluster_counts = []
#   for i, cluster in enumerate(sorted_clusters):
#       if cluster:
#           cluster_range = f"Cluster {i}: Range {min(cluster)} - {max(cluster)}"
#           cluster_count = len(cluster)
#       else:
#           cluster_range = f"Cluster {i}: Empty"
#           cluster_count = 0
#       sorted_cluster_ranges.append(cluster_range)
#       sorted_cluster_counts.append(cluster_count)
#       print(f"{cluster_range}, Count: {cluster_count}")
#   print('---------------------------------\n')
#
#
#   return sorted_labels.tolist(), sorted_clusters

# def range_prices(prices, num_whole_digits=2, decimal_places=2):
#   """
#   Converts prices to a specified range by scaling them by factors of 10
#   and rounds to a specified number of decimal places.

#   This is done to control/limit the number of unique prices,
#   thereby controlling the vocabulary size. This is helpful when dealing
#   with stocks priced in different price ranges.

#   Args:
#     prices: A list of float prices. Must be a list containing numeric types.
#     num_whole_digits: The desired number of whole digits for the ranged prices
#                       (e.g., 1 for ones, 2 for tens, etc.). Must be an integer (default: 2).
#     decimal_places: The desired number of decimal places for the ranged prices.
#                     Must be an integer greater than or equal to 0 (default: 2).

#   Returns:
#     A list of float prices that have been ranged and rounded.
#   """

#   # Input validation for prices
#   if not isinstance(prices, list):
#       raise TypeError("prices must be a list.")
#   for i, price in enumerate(prices):
#       if not isinstance(price, numbers.Number):
#           # Use IndexError to indicate the position of the problematic element
#           raise IndexError(f"Element at index {i} in 'prices' is not a number.")

#   # Input validation for num_whole_digits and decimal_places
#   if not isinstance(num_whole_digits, int):
#       raise TypeError("num_whole_digits must be an integer.")
#   if not isinstance(decimal_places, int):
#       raise TypeError("decimal_places must be an integer.")
#   if decimal_places < 0:
#       raise ValueError("decimal_places must be an integer greater than or equal to 0.")


#   ranged_prices = []

#   for price in prices:
#     if price == 0:
#       digits = 0
#     else:
#       digits = len(str(int(price)))

#     # Calculate the scaling factor
#     scaling_factor = 10**(digits - num_whole_digits)

#     # Apply scaling and rounding
#     scaled_price = round(price / scaling_factor, decimal_places)

#     # Correct prices that were rounded outside the intended range
#     if scaled_price >= 10**num_whole_digits:
#         scaled_price = 10**(num_whole_digits - 1)

#     ranged_prices.append(scaled_price)

#   return ranged_prices