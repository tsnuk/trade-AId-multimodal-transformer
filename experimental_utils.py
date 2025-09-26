"""experimental_utils.py

Experimental utility functions that are preserved for potential future use.

These functions are kept commented out because their inputs and interfaces
do not match the current code structure, but they contain potentially useful
logic that may be adapted for future features.

"""

# def kmeans_cluster_with_sorted_centers(data, n_clusters, outlier_threshold_percentile=0.1):
#   """
#   Apply KMeans clustering to data and sort clusters by minimum values.
#
#   Args:
#     data: List of numeric data points.
#     n_clusters: Number of clusters to create.
#     outlier_threshold_percentile: Percentile threshold for capping outliers (default: 0.1).
#
#   Returns:
#     sorted_labels: List of cluster labels sorted by cluster centers.
#     sorted_clusters: List of clusters sorted by center values.
#   """
#   capped_data = cap_outliers(data, outlier_threshold_percentile)
#
#   reshaped_data = np.array(capped_data).reshape(-1, 1)
#
#   kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#   kmeans.fit(reshaped_data)
#
#   labels = kmeans.labels_
#   centers = kmeans.cluster_centers_
#
#   sorted_indices = np.argsort(centers.flatten())
#
#   label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
#
#   sorted_labels = np.array([label_mapping[label] for label in labels])
#
#   sorted_clusters = [[] for _ in range(n_clusters)]
#   for i, label in enumerate(sorted_labels):
#       if i < len(capped_data):
#           sorted_clusters[label].append(capped_data[i])
#       else:
#           print(f"Warning: Index {i} out of bounds for data list in sorted_clusters creation.")
#
#
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
#   Convert prices to specified range by scaling and rounding.
#
#   Args:
#     prices: List of float prices.
#     num_whole_digits: Desired number of whole digits (default: 2).
#     decimal_places: Desired decimal places (default: 2).
#
#   Returns:
#     List of float prices that have been ranged and rounded.
#   """

#   if not isinstance(prices, list):
#       raise TypeError("prices must be a list.")
#   for i, price in enumerate(prices):
#       if not isinstance(price, numbers.Number):
#           raise IndexError(f"Element at index {i} in 'prices' is not a number.")

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

#     scaling_factor = 10**(digits - num_whole_digits)

#     scaled_price = round(price / scaling_factor, decimal_places)

#     if scaled_price >= 10**num_whole_digits:
#         scaled_price = 10**(num_whole_digits - 1)

#     ranged_prices.append(scaled_price)

#   return ranged_prices