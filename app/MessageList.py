import enum
import dateutil
import pandas as pd
import numpy as np
from flask import Blueprint
bp = Blueprint('MessageList', __name__)


def compare_messages_by_categorical_fields(msgs1, msgs2, field_names=None, verbose=True):
   result_array = []


   stat1, cat_field_names1 = msgs1.categorical_combinations_stat(field_names)
   stat2, cat_field_names2 = msgs2.categorical_combinations_stat(field_names)
   if cat_field_names1 != cat_field_names2:
       result_array.append({
           'error': f"Categorical columns in log files (and even their order) must be the same, but they are different:\n"
                    f"\tFields in \"{msgs1.filename}\": {cat_field_names1}\n"
                    f"\tFields in \"{msgs2.filename}\": {cat_field_names2}\n"
                    "(you can specify columns to compare manually to avoid this error)"
       })
       return result_array


   assert field_names is None or field_names == cat_field_names1
   if field_names is None:
       field_names = cat_field_names1
   cat_clusters1 = msgs1.categorical_clusters_from_stat(stat1, field_names)
   cat_clusters2 = msgs2.categorical_clusters_from_stat(stat2, field_names)


   result_array.append({'clusters1': cat_clusters1, 'clusters2': cat_clusters2})

   cluster_found = False
   file2_match_flags = [False]*len(cat_clusters2)
   for stat1 in cat_clusters1:
       cluster_count1 = len(stat1['cluster_combinations'])
       if cluster_count1 == 0:
           continue
       match_found = False
       for i, stat2 in enumerate(cat_clusters2):
           cluster_count2 = len(stat2['cluster_combinations'])
           if cluster_count2 == 0:
               continue
           if set([frozenset(comb) for comb in stat1['cluster_combinations']]) == set([frozenset(comb) for comb in stat2['cluster_combinations']]):
               match_found = True
               file2_match_flags[i] = True
               break
       if not cluster_found:
           result_array.append({'message': "The following clusters found:", 'clusters': []})
           cluster_found = True
       noun_count_str = "s" if cluster_count1 > 1 else ""
       verb_count_str = "" if cluster_count1 > 1 else "s"
       if match_found:
           cluster_result = {
               'message_instances': len(stat1['cluster_combinations']),
               'occurrence_percentage_msgs1': 100 * sum(stat1['cluster_vals']) / stat1['total_count'],
               'occurrence_percentage_msgs2': 100 * sum(stat2['cluster_vals']) / stat2['total_count'],
               'file1': msgs1.filename,
               'file2': msgs2.filename,
               'cluster_combinations': []
           }
           for i1, comb in enumerate(stat1['cluster_combinations']):
               instance_result = {
                   'percentage_msgs1': 100 * stat1['cluster_vals'][i1] / stat1['total_count'],
                   'percentage_msgs2': 100 * stat2['cluster_vals'][i1] / stat2['total_count'],
                   'values': [{field_names[j]: value} for j, value in enumerate(comb)]
               }
               cluster_result['cluster_combinations'].append(instance_result)
           result_array.append(cluster_result)
       else:
           no_match_result = {
               'message': f"\"{msgs1.filename}\" has the following {len(stat1['cluster_combinations'])} message instance{noun_count_str}, " +
                          f"that take{verb_count_str} {100 * sum(stat1['cluster_vals']) / stat1['total_count']:00.1f}% of its messages. " +
                          f"But \"{msgs2.filename}\" has no such frequent messages:",
               'instances': []
           }
           for i, comb in enumerate(stat1['cluster_combinations']):
               instance_result = {
                   'percentage_msgs1': 100 * stat1['cluster_vals'][i] / stat1['total_count'],
                   'values': [{field_names[j]: value} for j, value in enumerate(comb)]
               }
               no_match_result['instances'].append(instance_result)
           result_array.append(no_match_result)


   for i2, file2_match_flag in enumerate(file2_match_flags):
       if not file2_match_flag:
           stat2 = cat_clusters2[i2]
           cluster_count2 = len(stat2['cluster_combinations'])
           if cluster_count2 == 0:
               continue
           clusters_found = True
           noun_count_str = "s" if cluster_count2 > 1 else ""
           verb_count_str = "" if cluster_count2 > 1 else "s"
           no_match_result = {
               'message': f"\"{msgs2.filename}\" has the following {len(stat2['cluster_combinations'])} message instance{noun_count_str}, " +
                          f"that take{verb_count_str} {100 * sum(stat2['cluster_vals']) / stat2['total_count']:00.1f}% of its messages. " +
                          f"But \"{msgs1.filename}\" has no such frequent messages:",
               'instances': []
           }
           for i, comb in enumerate(stat2['cluster_combinations']):
               instance_result = {
                   'percentage_msgs2': 100 * stat2['cluster_vals'][i] / stat2['total_count'],
                   'values': [{field_names[j]: value} for j, value in enumerate(comb)]
               }
               no_match_result['instances'].append(instance_result)
           result_array.append(no_match_result)


   return result_array


def comparable_types(field1, field2):
   # TODO: implementation is missing
   return True


def align_base_field_types(field1, field2):
   # TODO: implementation is missing
   return field1, field2


FieldBaseType = enum.Enum('FieldBaseType', 'float integer categorical date string')


class FieldType_General:


   def __init__(self, data):
       if not isinstance(data, np.ndarray) or len(data.shape) != 1:
           raise ValueError(f"Expected 1-d numpy array, got: {type(data)}")
       self.data = data




class IntOrFloatMixin:


   def find_big_differences(self, other_field):
       BIG_DIFF_THR = 2  # We suppose that big diffs shold be greater than BIG_DIFF_THR*diff.mean()
       RARE_DIFF_THR = 0.1  # We suppose that rare diffs should occur less than RARE_DIFF_THR*len(data)
       MAX_THR_CHANGE_ITERS = 10  # No more than 10 iterations to find optimal threshold


       other_data = other_field.data
       if len(self.data) != len(other_data):
           raise ValueError(f"Lengths are not equal: {len(self.data)} and {len(other_data)}")
       diff = np.abs(self.data - other_data)
       if np.isclose(diff.mean(), 0) or diff.max() <= BIG_DIFF_THR*diff.mean():
           return np.array([], dtype=int)
       diff_vals = np.unique(diff)
       if len(diff_vals) < 2:
           return np.array([], dtype=int)
       biggest_thr = diff_vals[-2]  # one value before the maximum


       big_thr = BIG_DIFF_THR*diff.mean()
       if big_thr >= biggest_thr:
           return np.array([], dtype=int)
       found = False
       for thr in np.linspace(big_thr, biggest_thr, num=MAX_THR_CHANGE_ITERS):
           if len(np.where(diff > thr)) < RARE_DIFF_THR*len(diff):
               found = True
               break
       if found:
           times = np.argwhere(diff > thr)  # thr is still defined after loop
       else:
           times = np.array([], dtype=int)
       return times




class FieldType_Float(FieldType_General, IntOrFloatMixin):
   BASE_TYPE = FieldBaseType.float




class FieldType_Int(FieldType_General, IntOrFloatMixin):
   BASE_TYPE = FieldBaseType.integer




class FieldType_Cat(FieldType_General):
   BASE_TYPE = FieldBaseType.categorical


   def compare_categories(self, other_field):
       other_data = other_field.data
       if len(self.data) != len(other_data):
           raise ValueError(f"Lengths are not equal: {len(self.data)} and {len(other_data)}")
      
       diff = np.where(self.data == other_data, 0, 1)
       match_rel = np.count_nonzero(diff == 0)/len(diff)
       matched_vals = set(self.data[np.argwhere(diff == 0)].flatten())
       nonmatched_vals = set(self.data[np.argwhere(diff != 0)].flatten()) | set(other_data[np.argwhere(diff != 0)].flatten())
       strict_matched_vals = matched_vals - nonmatched_vals
       strict_nonmatched_vals = nonmatched_vals - matched_vals
       return match_rel, matched_vals, nonmatched_vals, strict_matched_vals, strict_nonmatched_vals


   def count_values(self):
       vals, counts = np.unique(self.data, return_counts=True)
       cnt_idxs = counts.argsort()
       return vals[cnt_idxs[::-1]], counts[cnt_idxs[::-1]]




class FieldType_Date(FieldType_General):
   BASE_TYPE = FieldBaseType.date


   def __init__(self, data):
       time_data = np.vectorize(lambda t: t.timestamp())(data)
       super().__init__(time_data)




class FieldType_Str(FieldType_General):
   BASE_TYPE = FieldBaseType.string




class FieldType_Resource(FieldType_Float):


   def __init__(self, data, low_val, high_val, low_warn_level, high_warn_level):
       super().__init__(data)
       self.low_val = low_val
       self.high_val = high_val
       self.low_warn_level = low_warn_level
       self.high_warn_level = high_warn_level




class FieldType_CPUUtilization(FieldType_Resource):


   def __init__(self, data, high_warn_level):
       super().__init__(data, 0, 100, None, high_warn_level)




class FieldType_RAMUtilization(FieldType_Resource):


   def __init__(self, data, low_warn_level, high_warn_level):
       super().__init__(data, 0, 100, low_warn_level, high_warn_level)




def compare_fields(field1: FieldType_General, field2: FieldType_General):
   field1, field2 = align_base_field_types(field1, field2)
   comparison_res = {}
   if issubclass(type(field1), IntOrFloatMixin):
       comparison_res['big_difference_idxs'] = field1.find_big_differences(field2)
   if issubclass(type(field1), FieldType_Cat):
       match_rel, matched_vals, nonmatched_vals, strict_matched_vals, strict_nonmatched_vals = field1.compare_categories(field2)
       comparison_res['matched_relation'] = match_rel
       comparison_res['matched_vals'] = matched_vals
       comparison_res['nonmatched_vals'] = nonmatched_vals
       comparison_res['strict_matched_vals'] = strict_matched_vals
       comparison_res['strict_nonmatched_vals'] = strict_nonmatched_vals
   return comparison_res


def create_field_object(field_s, name, verbose=True):
   obj = None
   if obj is None:
       if np.issubdtype(field_s.dtype, np.floating):
           obj = FieldType_Float(field_s.values)
       elif np.issubdtype(field_s.dtype, np.integer):
           # TODO: may be categorical?
           obj = FieldType_Int(field_s.values)
   if obj is None:
       # here we assume, that it is string, but it also can be categorical
       try:
           date_s = field_s.apply(dateutil.parser.parse)
           obj = FieldType_Date(date_s.values)
       except:
           pass
   if obj is None:
       try:
           float_s = field_s.apply(float)
           try:
               int_s = float_s.apply(int)
               obj = FieldType_Int(int_s.values)
           except:
               obj = FieldType_Float(float_s.values)
       except:
           pass
   if obj is None:
       if field_s.nunique() < 0.9*len(field_s):
           obj = FieldType_Cat(field_s.values)
       else:
           obj = FieldType_Str(field_s.values)
   if verbose:
       print(f"{name}: autodetected type is {obj.BASE_TYPE}")
   return obj




class MessageList:


   last_list_id = 1


   def __init__(self, init_df, filename, name="", verbose=True):
       self.fields = {col: create_field_object(init_df[col], col, verbose=verbose) for col in init_df.columns.tolist()}
       self.filename = filename
       if len(name) == 0:
           name = f"message_list_{MessageList.last_list_id}"
           MessageList.last_list_id += 1
       self.name = name


   def get_categorical_fields(self):
       return {f_name: f_field for f_name, f_field in self.fields.items() if isinstance(f_field, FieldType_Cat)}


   def categorical_combinations_stat(self, field_names=None):
       # same results can be obtained by Pandas "value_counts" dataframe function, but here we have more control on processing
       res_stat = {'combinations': None, 'counts': None, 'next_level': None}
       if field_names is None:
           field_names = list(self.get_categorical_fields().keys())
       if len(field_names) < 1:
           return res_stat


       current_stat = res_stat
       level_idxs = []
       level_values = []
       prev_stats = []
       current_level = 0
       going_forward = True
       all_idxs = np.arange(0, len(self.fields[field_names[0]].data), 1, dtype=int)
       while True:
           # print(f"combs: {current_level = } " + " ".join([f"{li}/{(len(lv) - 1)}" for li, lv in zip(level_idxs, level_values)]))
           if current_level > 0:
               msg_idx = all_idxs
               for i in range(0, current_level):
                   new_msg_idx = np.nonzero(self.fields[field_names[i]].data[msg_idx] == level_values[i][level_idxs[i]])
                   skipped_elems = np.setdiff1d(msg_idx, all_idxs, assume_unique=True)
                   for s_elem in np.sort(skipped_elems):
                       new_msg_idx[new_msg_idx > s_elem] += 1
                   msg_idx = new_msg_idx
               filtered_data = self.fields[field_names[current_level]].data[msg_idx]
           else:
               filtered_data = self.fields[field_names[current_level]].data


           # on the first pass data length must be > 0, but on the next passes it can be 0
           assert len(filtered_data) > 0 or current_level < len(level_idxs)


           if len(filtered_data) > 0:
               values, counts = np.unique(filtered_data, return_counts=True)
               cnt_idxs = counts.argsort()
               values = values[cnt_idxs[::-1]].tolist()
               counts = counts[cnt_idxs[::-1]].tolist()
  
               if len(level_idxs) <= current_level:
                   level_idxs.append(0)
                   level_values.append(values)
                   assert len(level_idxs) == current_level + 1
  
                   hist_list = [level_values[i][level_idxs[i]] for i in range(0, current_level)]
                   current_stat['combinations'] = [hist_list + [v] for v in values]
                   current_stat['counts'] = counts
               else:
                   hist_list = [level_values[i][level_idxs[i]] for i in range(0, current_level)]
                   assert hist_list + [values[0]] not in current_stat['combinations']
                   current_stat['combinations'] += [hist_list + [v] for v in values]
                   current_stat['counts'] += counts


           if current_level == len(field_names) - 1:
               going_forward = False


           if not going_forward:
               level_idxs[current_level] += len(level_values[current_level])  # we see all items at the last level
               exit_flag = False
               while level_idxs[current_level] > len(level_values[current_level]) - 1:
                   if current_level == 0:
                       exit_flag = True
                       break
                   level_idxs[current_level] = 0
                   level_idxs[current_level - 1] += 1
                   current_level -= 1
                   current_stat = prev_stats.pop(-1)
               if exit_flag:
                   break
               going_forward = True


           current_level += 1
           if len(level_idxs) <= current_level:
               current_stat['next_level'] = {'combinations': None, 'counts': None, 'next_level': None}
           prev_stats.append(current_stat)
           current_stat = current_stat['next_level']


       return res_stat, field_names


   def categorical_clusters_from_stat(self, categorical_stat, field_names):
       MAX_CLUSTER_COUNT = 4
       CLUSTERS_MIN_PERCENT = 0.7


       found_clusters = []
       current_stat = categorical_stat
       for _ in range(len(field_names)):
           level_stat = {'cluster_combinations': [], 'cluster_vals': [], 'noise_combinations': [], 'noise_vals': [], 'total_count': None}


           assert len(set(tuple(l) for l in current_stat['combinations'])) == len(current_stat['combinations'])
           if len(current_stat['counts']) > 1:
               cluster_probs = np.array(current_stat['counts']) / sum(current_stat['counts'])
               cluster_idxs = np.argsort(cluster_probs)[::-1]
               # subtract sum of smallest clusters from sum of the biggest ones
               # (last element is not needed, because in this case either smallest or biggest cluster is absent)
               big_clusters_sum = np.cumsum(cluster_probs[cluster_idxs[:-1]])
               small_clusters_sum = np.cumsum(cluster_probs[cluster_idxs[::-1][:-1]])[::-1]
               clusters_percent = big_clusters_sum - small_clusters_sum
               big_clusters_idxs = np.argwhere(clusters_percent[:MAX_CLUSTER_COUNT] >= CLUSTERS_MIN_PERCENT).flatten()
               if len(big_clusters_idxs) > 0:
                   # since we search for the smallest combination of biggest clusters, we take only first index of
                   # big_clusters_idxs, which corresponds to the sum of precents of biggest clusters
                   found_cluster_idxs = cluster_idxs[:big_clusters_idxs[0] + 1]
               else:
                   # corresponds to big amount of rare enough combinations
                   found_cluster_idxs = np.array([])
           else:
               # single cluster is not interesting
               found_cluster_idxs = np.array([])


           assert len(current_stat['counts']) == len(current_stat['combinations'])
           assert all([len(c) == _ + 1 for c in current_stat['combinations']])
           level_stat['cluster_combinations'] = [comb for i, comb in enumerate(current_stat['combinations']) if i in found_cluster_idxs]
           level_stat['cluster_vals'] = [cnt for i, cnt in enumerate(current_stat['counts']) if i in found_cluster_idxs]
           level_stat['noise_combinations'] = [comb for i, comb in enumerate(current_stat['combinations']) if i not in found_cluster_idxs]
           level_stat['noise_vals'] = [cnt for i, cnt in enumerate(current_stat['counts']) if i not in found_cluster_idxs]
           level_stat['total_count'] = sum(current_stat['counts'])
           found_clusters.append(level_stat)


           current_stat = current_stat['next_level']


       return found_clusters

   def convert_categorical_clusters_to_json(self, found_cluster_stat, field_names):
        MAX_NOISE_COMBS = 3
        result_json = {'clusters': []}
        cluster_found = False

        for stat in found_cluster_stat:
            cluster_count = len(stat['cluster_combinations'])
            if cluster_count == 0:
                continue

            cluster_instances = []
            for i, comb in enumerate(stat['cluster_combinations']):
                instance_values = [{field_names[j]: value} for j, value in enumerate(comb)]
                occurrence_percentage = 100 * stat['cluster_vals'][i] / stat['total_count']
                instance_result = {
                    'fields': instance_values,
                    'occurrence_percentage': occurrence_percentage,
                }
                cluster_instances.append(instance_result)

            cluster_result = {
                'message_instances': cluster_instances,
                'occurrence_percentage': 100 * sum(stat['cluster_vals']) / stat['total_count'],
            }

            result_json['clusters'].append(cluster_result)

            noise_instances_count = len(stat['noise_combinations'])
            if noise_instances_count > 0:
                biggest_combs_idxs = np.argsort(stat['noise_vals'])[::-1][:MAX_NOISE_COMBS]
                sorted_noise_combs = list(np.array(stat['noise_combinations'])[biggest_combs_idxs])
                sorted_noise_vals = list(np.array(stat['noise_vals'])[biggest_combs_idxs])

                noise_instances = []
                for i, (comb, count) in enumerate(zip(sorted_noise_combs, sorted_noise_vals)):
                    instance_values = [{field_names[j]: value} for j, value in enumerate(comb)]
                    occurrence_percentage = 100 * count / stat['total_count']
                    instance_result = {
                        'fields': instance_values,
                        'percentage': occurrence_percentage,
                    }
                    noise_instances.append(instance_result)

                noise_result = {
                    'noise_instances': noise_instances,
                    'occurrence_percentage': 100 * sum(stat['noise_vals']) / stat['total_count'],
                }

                result_json['clusters'].append(noise_result)

                cluster_found = True

        if not cluster_found:
            result_json['message'] = "No clusters found"

        return result_json





"""def main():
   print("Python script is being executed.")
   file1_path = 'src/scripts/output_file1.csv'
   file2_path = 'src/scripts/output_file2.csv'


   try:
       # Read CSV files using pandas
       logs1_df = pd.read_csv(file1_path)
       logs2_df= pd.read_csv(file2_path)
       msgs1 =  MessageList(logs1_df,file1_path, verbose=True)
       msgs2 =  MessageList(logs2_df,file2_path, verbose=True)
       stat1, cat_field_names1 = msgs1.categorical_combinations_stat()
       print(cat_field_names1)
       stat2, cat_field_names2 = msgs2.categorical_combinations_stat()
       print(cat_field_names2)
       cat_clusters1=msgs1.categorical_clusters_from_stat(stat1, cat_field_names1)
       cat_clusters2=msgs2.categorical_clusters_from_stat(stat2, cat_field_names2)
      
       msgs1.print_categorical_clusters(cat_clusters1, cat_field_names1)
       msgs2.print_categorical_clusters(cat_clusters2, cat_field_names2)
 
       res= compare_messages_by_categorical_fields(msgs1, msgs2, field_names=['level', 'method'])
       print(res)






   except Exception as e:
       print(f"Error reading CSV files: {e}")
       raise  # Re-raise the exception to be caught by the calling code


if __name__ == "__main__":
   main()
"""

