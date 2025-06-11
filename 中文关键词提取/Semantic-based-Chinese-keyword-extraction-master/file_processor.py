import os
from collections import defaultdict
import textPrecessing
import intermediate
import statistics
import outPut
import json

class FileProcessor:
    def __init__(self):
        self.cache = {}  # 用于缓存处理结果
        self.use_simple_preprocess = True  # 默认使用简化预处理

    def process_file(self, file_path, stage):
        """处理单个文件"""
        # 获取文件名和路径
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_dir = os.path.dirname(file_path)
        output_dir = os.path.normpath(os.path.join(file_dir, file_name + "_output"))

        # 创建输出目录
        for subdir in ["预处理", "语义特征", "统计特征", "词语得分"]:
            subdir_path = os.path.normpath(os.path.join(output_dir, subdir))
            os.makedirs(subdir_path, exist_ok=True)

        # 读取文件内容
        title, body = self._read_file(file_path)
        text = title + "。" + body

        # 根据处理阶段执行相应的操作
        if stage == "预处理":
            return self._preprocess(text, title, output_dir)
        elif stage == "提取语义特征":
            return self._extract_semantic_features(text, title, output_dir)
        elif stage == "提取统计特征":
            return self._extract_statistical_features(text, title, output_dir)
        elif stage == "计算词语得分":
            return self._calculate_scores(text, title, output_dir)
        else:
            raise ValueError(f"未知的处理阶段: {stage}")

    def _read_file(self, file_path):
        """读取文件内容"""
        try:
            # 使用uploadFile模块读取文件
            import uploadFile
            title, body = uploadFile.readFile(file_path)
            return title, body
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")
            # 尝试直接读取文件
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 假设第一行是标题，其余是正文
                    lines = content.split('\n')
                    title = lines[0] if lines else ""
                    body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                    return title, body
            except Exception as e2:
                print(f"备用读取方式也失败: {str(e2)}")
                return "", ""

    def _preprocess(self, text, title, output_dir):
        """预处理阶段"""
        try:
            print(f"使用{'简化' if self.use_simple_preprocess else '完整'}预处理方式")
            
            # 确保输入是字符串类型
            if not isinstance(text, str):
                print(f"警告: 文本不是字符串类型: {type(text)}")
                text = str(text) if text is not None else ""
                
            if not isinstance(title, str):
                print(f"警告: 标题不是字符串类型: {type(title)}")
                title = str(title) if title is not None else ""
            
            # 分词和词性标注
            body = text.replace(title + "。", "", 1) if title else text  # 从文本中去除标题部分
            
            if self.use_simple_preprocess:
                # 使用简化版预处理模块
                try:
                    import simple_preprocessor
                    
                    # 调用简化预处理
                    result = simple_preprocessor.simple_preprocess(body, title)
                    
                    # 确保返回的是完整的元组
                    if not result or len(result) != 8:
                        print(f"警告: 简化预处理返回的结果不完整: 预期8个元素，实际{len(result) if result else 0}个")
                        # 创建默认值
                        wordsData, wordsFlagDict = [], {}
                        firstSentence, lastSentence = "", ""
                        nextDict, nextWordSum = {}, defaultdict(int)
                        preDict, preWordSum = {}, defaultdict(int)
                    else:
                        # 解包返回值
                        wordsData, wordsFlagDict, firstSentence, lastSentence, nextDict, nextWordSum, preDict, preWordSum = result
                        
                        # 确保各个返回值是正确的类型
                        if not isinstance(wordsData, list):
                            print(f"警告: wordsData 不是列表类型: {type(wordsData)}")
                            wordsData = list(wordsData) if hasattr(wordsData, '__iter__') else []
                            
                        if not isinstance(wordsFlagDict, dict):
                            print(f"警告: wordsFlagDict 不是字典类型: {type(wordsFlagDict)}")
                            wordsFlagDict = dict(wordsFlagDict) if hasattr(wordsFlagDict, 'items') else {}
                            
                        if not isinstance(nextDict, dict):
                            print(f"警告: nextDict 不是字典类型: {type(nextDict)}")
                            nextDict = dict(nextDict) if hasattr(nextDict, 'items') else {}
                            
                        if not isinstance(preDict, dict):
                            print(f"警告: preDict 不是字典类型: {type(preDict)}")
                            preDict = dict(preDict) if hasattr(preDict, 'items') else {}
                            
                except ImportError as ie:
                    print(f"导入简化预处理模块失败: {str(ie)}")
                    # 使用默认值
                    wordsData, wordsFlagDict = [], {}
                    firstSentence, lastSentence = "", ""
                    nextDict, nextWordSum = {}, defaultdict(int)
                    preDict, preWordSum = {}, defaultdict(int)
                except Exception as e:
                    print(f"调用简化预处理时出错: {str(e)}")
                    # 使用默认值
                    wordsData, wordsFlagDict = [], {}
                    firstSentence, lastSentence = "", ""
                    nextDict, nextWordSum = {}, defaultdict(int)
                    preDict, preWordSum = {}, defaultdict(int)
            else:
                # 使用原始预处理模块
                try:
                    import textPrecessing
                    
                    # 调用原始预处理
                    result = textPrecessing.word_segmentation(body, title)
                    
                    # 确保返回的是完整的元组
                    if not result or len(result) != 8:
                        print(f"警告: 原始预处理返回的结果不完整: 预期8个元素，实际{len(result) if result else 0}个")
                        # 创建默认值
                        wordsData, wordsFlagDict = [], {}
                        firstSentence, lastSentence = "", ""
                        nextDict, nextWordSum = {}, defaultdict(int)
                        preDict, preWordSum = {}, defaultdict(int)
                    else:
                        # 解包返回值
                        wordsData, wordsFlagDict, firstSentence, lastSentence, nextDict, nextWordSum, preDict, preWordSum = result
                except ImportError as ie:
                    print(f"导入原始预处理模块失败: {str(ie)}")
                    # 使用默认值
                    wordsData, wordsFlagDict = [], {}
                    firstSentence, lastSentence = "", ""
                    nextDict, nextWordSum = {}, defaultdict(int)
                    preDict, preWordSum = {}, defaultdict(int)
                except Exception as e:
                    print(f"调用原始预处理时出错: {str(e)}")
                    # 使用默认值
                    wordsData, wordsFlagDict = [], {}
                    firstSentence, lastSentence = "", ""
                    nextDict, nextWordSum = {}, defaultdict(int)
                    preDict, preWordSum = {}, defaultdict(int)
                
            # 保存预处理结果
            self._save_preprocess_results(wordsData, wordsFlagDict, firstSentence, lastSentence, 
                                        nextDict, nextWordSum, preDict, preWordSum, output_dir)

            return wordsData, wordsFlagDict, firstSentence, lastSentence, nextDict, nextWordSum, preDict, preWordSum
        except Exception as e:
            print(f"预处理调用出错: {str(e)}")
            # 出错时返回空数据结构
            return [], {}, "", "", {}, defaultdict(int), {}, defaultdict(int)

    def _extract_semantic_features(self, text, title, output_dir):
        """提取语义特征阶段"""
        # 从缓存或文件加载预处理结果
        wordsData, wordsFlagDict, firstSentence, lastSentence, nextDict, nextWordSum, preDict, preWordSum = \
            self._load_preprocess_results(output_dir)

        # 计算语义密度
        interDensity = intermediate.getDensity(wordsData)

        # 处理前后继关系
        interDensity, wordsData, wordsFlagDict = self._add_word(interDensity, wordsData, wordsFlagDict,
                                                              nextDict, nextWordSum, preDict, preWordSum)

        # 保存语义特征
        self._save_semantic_features(interDensity, output_dir)

        return interDensity, wordsData, wordsFlagDict

    def _extract_statistical_features(self, text, title, output_dir):
        """提取统计特征阶段"""
        try:
            print("开始提取统计特征...")
            
            # 从缓存或文件加载预处理结果
            wordsData, wordsFlagDict, firstSentence, lastSentence, nextDict, nextWordSum, preDict, preWordSum = \
                self._load_preprocess_results(output_dir)
                
            # 检查预处理结果是否有效
            if not wordsData:
                print("警告: 预处理结果为空，无法提取统计特征")
                return {}, {}, {}, {}
                
            print(f"成功加载预处理结果，共有 {len(wordsData)} 个候选词")
            
            # 创建统计特征输出目录
            statistical_dir = os.path.normpath(os.path.join(output_dir, "统计特征"))
            os.makedirs(statistical_dir, exist_ok=True)
            
            # 计算TF-IDF特征
            print("计算TF-IDF特征...")
            try:
                import statistics
                wordsTfidf = statistics.getTfidf(len(wordsData), text)
                
                # 确保所有候选词都有TF-IDF值
                for word in wordsData:
                    if word not in wordsTfidf:
                        wordsTfidf[word] = 0.5  # 默认中间值
                
                # 归一化TF-IDF特征
                wordsTfidf = self._normalize(wordsData, wordsTfidf)
                
                # 保存TF-IDF特征
                self._save_feature(wordsTfidf, "wordsTfidf", statistical_dir)
                print(f"成功计算TF-IDF特征，共有 {len(wordsTfidf)} 个词的TF-IDF值")
            except Exception as e:
                print(f"计算TF-IDF特征时出错: {str(e)}")
                # 创建默认TF-IDF特征
                wordsTfidf = {}
                for word in wordsData:
                    wordsTfidf[word] = 0.5  # 默认中间值
                print("使用默认值作为TF-IDF特征")
            
            # 计算位置特征
            print("计算位置特征...")
            try:
                wordsLoc = statistics.getLoc(wordsData, wordsTfidf, title, firstSentence, lastSentence)
                
                # 确保所有候选词都有位置特征
                for word in wordsData:
                    if word not in wordsLoc:
                        wordsLoc[word] = 0.5  # 默认中间值
                
                # 归一化位置特征
                wordsLoc = self._normalize(wordsData, wordsLoc)
                
                # 保存位置特征
                self._save_feature(wordsLoc, "wordsLoc", statistical_dir)
                print(f"成功计算位置特征，共有 {len(wordsLoc)} 个词的位置特征")
            except Exception as e:
                print(f"计算位置特征时出错: {str(e)}")
                # 创建默认位置特征
                wordsLoc = {}
                for word in wordsData:
                    wordsLoc[word] = 0.5  # 默认中间值
                print("使用默认值作为位置特征")
            
            # 计算词性特征
            print("计算词性特征...")
            try:
                wordsFlagWeight = statistics.getFlag(wordsFlagDict, wordsData)
                
                # 确保所有候选词都有词性特征
                for word in wordsData:
                    if word not in wordsFlagWeight:
                        wordsFlagWeight[word] = 0.5  # 默认中间值
                
                # 归一化词性特征
                wordsFlagWeight = self._normalize(wordsData, wordsFlagWeight)
                
                # 保存词性特征
                self._save_feature(wordsFlagWeight, "wordsFlagWeight", statistical_dir)
                print(f"成功计算词性特征，共有 {len(wordsFlagWeight)} 个词的词性特征")
            except Exception as e:
                print(f"计算词性特征时出错: {str(e)}")
                # 创建默认词性特征
                wordsFlagWeight = {}
                for word in wordsData:
                    wordsFlagWeight[word] = 0.5  # 默认中间值
                print("使用默认值作为词性特征")
            
            # 计算长度特征（可选的附加特征）
            print("计算长度特征...")
            try:
                wordsLen = {}
                for word in wordsData:
                    wordsLen[word] = len(word) / 10.0  # 按词长归一化，最长设为10个字符
                    if wordsLen[word] > 1.0:
                        wordsLen[word] = 1.0
                
                # 保存长度特征
                self._save_feature(wordsLen, "wordsLen", statistical_dir)
                print(f"成功计算长度特征，共有 {len(wordsLen)} 个词的长度特征")
            except Exception as e:
                print(f"计算长度特征时出错: {str(e)}")
                # 创建默认长度特征
                wordsLen = {}
                for word in wordsData:
                    wordsLen[word] = 0.5  # 默认中间值
                print("使用默认值作为长度特征")
            
            print("统计特征提取完成")
            return wordsTfidf, wordsLoc, wordsFlagWeight, wordsLen
        except Exception as e:
            print(f"提取统计特征时出错: {str(e)}")
            # 出错时返回空字典
            return {}, {}, {}, {}

    def _calculate_scores(self, text, title, output_dir):
        """计算词语得分阶段"""
        try:
            # 从缓存或文件加载特征
            interDensity = self._load_semantic_features(output_dir)
            wordsTfidf, wordsLoc, wordsFlagWeight, wordsLen = self._load_statistical_features(output_dir)
            wordsData, _, _, _, _, _, _, _ = self._load_preprocess_results(output_dir)
            
            # 检查是否有足够的数据
            if not wordsData:
                print("警告: 没有候选词，无法计算得分")
                return {}, []
                
            if not interDensity:
                print("警告: 缺少语义特征，使用默认值")
                interDensity = {word: 0.5 for word in wordsData}
                
            if not wordsTfidf:
                print("警告: 缺少TF-IDF特征，使用默认值")
                wordsTfidf = {word: 0.5 for word in wordsData}
                
            if not wordsLoc:
                print("警告: 缺少位置特征，使用默认值")
                wordsLoc = {word: 0.5 for word in wordsData}
                
            if not wordsFlagWeight:
                print("警告: 缺少词性特征，使用默认值")
                wordsFlagWeight = {word: 0.5 for word in wordsData}
                
            print(f"开始计算 {len(wordsData)} 个候选词的得分...")
            
            # 计算词语得分
            score = self._calculate_score(wordsData, interDensity, wordsLoc, wordsTfidf, wordsFlagWeight)
            
            # 对得分进行排序，选出得分最高的词语作为关键词
            sorted_words = sorted(score.items(), key=lambda x: x[1], reverse=True)
            
            # 取前20个词作为关键词，或者得分大于0.5的词
            keywords = [item for item in sorted_words if item[1] > 0.1][:20]
            
            # 保存得分和关键词
            self._save_scores(score, output_dir)
            self._save_keywords(keywords, output_dir)
            
            print(f"得分计算完成，提取了 {len(keywords)} 个关键词")
            
            return score, keywords
        except Exception as e:
            print(f"计算词语得分时出错: {str(e)}")
            return {}, []

    def _normalize(self, wordsData, dict_obj):
        """归一化特征值"""
        try:
            # 确保字典是字典类型
            if not isinstance(dict_obj, dict):
                print(f"警告: 待归一化的对象不是字典类型: {type(dict_obj)}")
                # 如果不是字典但可以转换为字典，尝试转换
                if hasattr(dict_obj, 'items'):
                    dict_obj = dict(dict_obj)
                else:
                    # 无法处理，返回空字典
                    print("归一化失败: 无法将输入转换为字典")
                    return {}
                
            # 确保字典非空且有效值
            if not dict_obj:
                print("归一化失败: 输入字典为空")
                return dict_obj
                
            # 确保wordsData是列表
            if not isinstance(wordsData, list):
                print(f"警告: wordsData不是列表类型: {type(wordsData)}")
                if hasattr(wordsData, '__iter__') and not isinstance(wordsData, str):
                    wordsData = list(wordsData)
                else:
                    print("归一化失败: 无法将wordsData转换为列表")
                    return dict_obj
                    
            # 创建有效的字典 - 只包含wordsData中存在的词
            valid_dict = {}
            for word in wordsData:
                if word in dict_obj:
                    try:
                        # 确保值是数值类型
                        value = dict_obj[word]
                        if isinstance(value, (int, float)):
                            valid_dict[word] = value
                        else:
                            try:
                                # 尝试转换为浮点数
                                valid_dict[word] = float(value)
                            except (ValueError, TypeError):
                                print(f"警告: 词'{word}'的值'{value}'无法转换为数值，跳过")
                    except Exception as e:
                        print(f"处理词'{word}'时出错: {str(e)}")
            
            # 如果有效字典为空，返回原字典
            if not valid_dict:
                print("归一化失败: 没有有效的数值可以归一化")
                return dict_obj
            
            # 获取最大最小值
            try:
                min_word = min(valid_dict, key=valid_dict.get)
                min_value = valid_dict[min_word]
                max_word = max(valid_dict, key=valid_dict.get)
                max_value = valid_dict[max_word]
                
                # 输出调试信息
                print(f"归一化范围: 最小值 = {min_value}，最大值 = {max_value}")
            except ValueError as ve:
                print(f"归一化时出错 (可能是空字典): {str(ve)}")
                return dict_obj
            except Exception as e:
                print(f"获取最大最小值时出错: {str(e)}")
                return dict_obj
            
            # 确保分母不为0
            diff = max_value - min_value
            if abs(diff) < 1e-10:  # 使用小数比较代替直接等于0
                print("归一化失败: 最大值与最小值相等，所有值设为0.5")
                # 如果所有值相同，返回全为0.5的字典
                for word in wordsData:
                    if word in dict_obj:
                        dict_obj[word] = 0.5
                return dict_obj
            
            # 正常归一化
            for word in wordsData:
                if word in dict_obj:
                    try:
                        # 安全获取原始值
                        orig_value = dict_obj[word]
                        # 确保原始值是数值类型
                        if not isinstance(orig_value, (int, float)):
                            try:
                                orig_value = float(orig_value)
                            except (ValueError, TypeError):
                                print(f"警告: 词'{word}'的值'{orig_value}'无法转换为数值，设为0.5")
                                dict_obj[word] = 0.5
                                continue
                        
                        # 执行归一化
                        dict_obj[word] = (orig_value - min_value) / diff
                    except Exception as e:
                        print(f"归一化词'{word}'时出错: {str(e)}，设为0.5")
                        dict_obj[word] = 0.5
            
            return dict_obj
        except Exception as e:
            print(f"归一化特征时出错: {str(e)}")
            # 出错返回原字典
            return dict_obj

    def _calculate_score(self, wordsData, interDensity, wordsLoc, wordsTfidf, wordsFlagWeight):
        """计算词语最终得分"""
        score = defaultdict(float)
        
        try:
            # 确保所有输入都是有效的字典类型
            if not isinstance(interDensity, dict) or not isinstance(wordsLoc, dict) or \
               not isinstance(wordsTfidf, dict) or not isinstance(wordsFlagWeight, dict):
                print("警告: 计算得分的输入不都是字典类型")
                
                # 尝试将非字典类型转换为字典
                if hasattr(interDensity, 'items') and not isinstance(interDensity, dict):
                    interDensity = dict(interDensity)
                if hasattr(wordsLoc, 'items') and not isinstance(wordsLoc, dict):
                    wordsLoc = dict(wordsLoc)
                if hasattr(wordsTfidf, 'items') and not isinstance(wordsTfidf, dict):
                    wordsTfidf = dict(wordsTfidf)
                if hasattr(wordsFlagWeight, 'items') and not isinstance(wordsFlagWeight, dict):
                    wordsFlagWeight = dict(wordsFlagWeight)
            
            # 确保wordsData是列表
            if not isinstance(wordsData, list):
                print(f"警告: wordsData不是列表类型: {type(wordsData)}")
                if hasattr(wordsData, '__iter__') and not isinstance(wordsData, str):
                    wordsData = list(wordsData)
                else:
                    print("计算得分失败: 无法将wordsData转换为列表")
                    return score
            
            # 检查词语是否在所有特征字典中都有值
            valid_words = []
            for word in wordsData:
                if (word in interDensity or word in wordsLoc or 
                    word in wordsTfidf or word in wordsFlagWeight):
                    valid_words.append(word)
            
            if not valid_words:
                print("警告: 没有有效的词语可以计算得分")
                return score
                
            # 定义权重参数
            sW = 0.4  # 降低居间度权重
            Tw = 0.6  # 提高统计特征权重
            posW = 0.3  # 降低词性权重
            locW = 0.8  # 降低位置权重
            tfidfW = 1.2  # 提高TF-IDF权重

            total_processed = 0
            total_errors = 0
            for word in valid_words:
                try:
                    # 安全获取各个特征值，默认为0
                    inter_density_value = 0.0
                    if word in interDensity:
                        if isinstance(interDensity[word], (int, float)):
                            inter_density_value = interDensity[word]
                        else:
                            try:
                                inter_density_value = float(interDensity[word])
                            except (ValueError, TypeError):
                                print(f"警告: 词'{word}'的居间度值不是数值类型")
                    
                    words_flag_weight_value = 0.0
                    if word in wordsFlagWeight:
                        if isinstance(wordsFlagWeight[word], (int, float)):
                            words_flag_weight_value = wordsFlagWeight[word]
                        else:
                            try:
                                words_flag_weight_value = float(wordsFlagWeight[word])
                            except (ValueError, TypeError):
                                print(f"警告: 词'{word}'的词性权重值不是数值类型")
                    
                    words_loc_value = 0.0
                    if word in wordsLoc:
                        if isinstance(wordsLoc[word], (int, float)):
                            words_loc_value = wordsLoc[word]
                        else:
                            try:
                                words_loc_value = float(wordsLoc[word])
                            except (ValueError, TypeError):
                                print(f"警告: 词'{word}'的位置值不是数值类型")
                    
                    words_tfidf_value = 0.0
                    if word in wordsTfidf:
                        if isinstance(wordsTfidf[word], (int, float)):
                            words_tfidf_value = wordsTfidf[word]
                        else:
                            try:
                                words_tfidf_value = float(wordsTfidf[word])
                            except (ValueError, TypeError):
                                print(f"警告: 词'{word}'的TFIDF值不是数值类型")
                    
                    # 使用非线性组合方式计算得分
                    statistical_part = (posW * words_flag_weight_value + 
                                       locW * words_loc_value + 
                                       tfidfW * words_tfidf_value)
                    
                    location_factor = 1 + 0.2 * words_loc_value
                    
                    score[word] = (sW * inter_density_value + Tw * statistical_part) * location_factor
                    total_processed += 1
                except Exception as e:
                    print(f"计算词语'{word}'得分时出错: {str(e)}")
                    score[word] = 0.0
                    total_errors += 1
            
            print(f"成功计算了 {total_processed} 个词的得分，有 {total_errors} 个词计算出错")
        except Exception as e:
            print(f"计算词语得分时出错: {str(e)}")
            
        return score

    def _add_word(self, interDensity, wordsData, wordsFlagDict, nextDict, nextWordSum, preDict, preWordSum):
        """处理前后继关系"""
        try:
            for word1 in nextDict:
                # 确保nextWordSum[word1]是数值类型
                if isinstance(nextWordSum, dict) and nextWordSum.get(word1, 0) > 2:
                    # 确保nextDict[word1]是字典类型
                    word1_dict = nextDict.get(word1, {})
                    if not isinstance(word1_dict, dict):
                        # 如果不是字典但有items方法，尝试转换
                        if hasattr(word1_dict, 'items'):
                            word1_dict = dict(word1_dict)
                        else:
                            # 如果既不是字典也没有items方法，跳过此项
                            continue
                    
                    for word2 in word1_dict:
                        # 确保preDict[word2]存在且是字典类型
                        if word2 in preDict:
                            word2_dict = preDict.get(word2, {})
                            if not isinstance(word2_dict, dict):
                                # 如果不是字典但有items方法，尝试转换
                                if hasattr(word2_dict, 'items'):
                                    word2_dict = dict(word2_dict)
                                else:
                                    # 如果既不是字典也没有items方法，跳过此项
                                    continue
                            
                            if word1 in word2_dict:
                                # 安全获取值
                                next_ratio = word1_dict.get(word2, 0) / nextWordSum.get(word1, 1)
                                pre_ratio = word2_dict.get(word1, 0) / preWordSum.get(word2, 1)
                                
                                if next_ratio >= 0.8 and pre_ratio >= 0.8:
                                    strs = word1 + word2
                                    wordsData.append(strs)
                                    interDensity[strs] = max(interDensity.get(word1, 0), interDensity.get(word2, 0))
                                    wordsFlagDict[strs] = wordsFlagDict.get(word2, "n")  # 默认为名词
        except Exception as e:
            print(f"处理前后继关系时出错: {str(e)}")
            
        return interDensity, wordsData, wordsFlagDict

    # 文件保存和加载方法
    def _save_preprocess_results(self, wordsData, wordsFlagDict, firstSentence, lastSentence, 
                                nextDict, nextWordSum, preDict, preWordSum, output_dir):
        """保存预处理结果"""
        # 规范化路径
        preprocess_dir = os.path.normpath(os.path.join(output_dir, "预处理"))
        
        # 确保目录存在
        os.makedirs(preprocess_dir, exist_ok=True)
        
        try:
            # 确保wordsData是列表
            if not isinstance(wordsData, list):
                print(f"警告: wordsData不是列表类型: {type(wordsData)}")
                wordsData = list(wordsData) if hasattr(wordsData, '__iter__') and not isinstance(wordsData, str) else []
            
            # 保存wordsData
            words_data_path = os.path.join(preprocess_dir, "wordsData.txt")
            with open(words_data_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(str(word) for word in wordsData))

            # 确保wordsFlagDict是字典
            if not isinstance(wordsFlagDict, dict):
                print(f"警告: wordsFlagDict不是字典类型: {type(wordsFlagDict)}")
                wordsFlagDict = dict(wordsFlagDict) if hasattr(wordsFlagDict, 'items') else {}
                
            # 保存wordsFlagDict
            words_flag_txt_path = os.path.join(preprocess_dir, "wordsFlagDict.txt")
            words_flag_json_path = os.path.join(preprocess_dir, "wordsFlagDict.json")
            
            try:
                outPut.writeDict(words_flag_txt_path, wordsFlagDict)
                outPut.writeDictToJson(words_flag_json_path, wordsFlagDict, 'wordsFlagDict')
            except Exception as e:
                print(f"保存wordsFlagDict时出错: {str(e)}")
                # 直接写入简单的文本文件作为备份
                with open(words_flag_txt_path, 'w', encoding='utf-8') as f:
                    for key, value in wordsFlagDict.items():
                        f.write(f"{key}: {value}\n")
                        
            # 保存前后继词信息 - 将嵌套字典转换为可序列化的形式
            # 处理嵌套字典时使用自定义方法防止序列化错误
            try:
                # 确保nextDict和preDict是字典
                if not isinstance(nextDict, dict):
                    print(f"警告: nextDict不是字典类型: {type(nextDict)}")
                    nextDict = dict(nextDict) if hasattr(nextDict, 'items') else {}
                    
                if not isinstance(preDict, dict):
                    print(f"警告: preDict不是字典类型: {type(preDict)}")
                    preDict = dict(preDict) if hasattr(preDict, 'items') else {}
                
                # 准备序列化的数据
                next_dict_serialized = {}
                for key, value in nextDict.items():
                    if isinstance(value, dict) or hasattr(value, 'items'):
                        next_dict_serialized[key] = dict(value)
                    else:
                        next_dict_serialized[key] = {}
                    
                pre_dict_serialized = {}
                for key, value in preDict.items():
                    if isinstance(value, dict) or hasattr(value, 'items'):
                        pre_dict_serialized[key] = dict(value)
                    else:
                        pre_dict_serialized[key] = {}
                
                # 路径
                next_dict_path = os.path.join(preprocess_dir, "nextDict.json")
                next_word_sum_path = os.path.join(preprocess_dir, "nextWordSum.json")
                pre_dict_path = os.path.join(preprocess_dir, "preDict.json")
                pre_word_sum_path = os.path.join(preprocess_dir, "preWordSum.json")
                
                # 准备nextWordSum和preWordSum
                next_word_sum_dict = dict(nextWordSum) if hasattr(nextWordSum, 'items') else {}
                pre_word_sum_dict = dict(preWordSum) if hasattr(preWordSum, 'items') else {}
                
                # 保存
                with open(next_dict_path, 'w', encoding='utf-8') as f:
                    json.dump({'nextDict': next_dict_serialized}, f, ensure_ascii=False, indent=2)
                    
                with open(next_word_sum_path, 'w', encoding='utf-8') as f:
                    json.dump({'nextWordSum': next_word_sum_dict}, f, ensure_ascii=False, indent=2)
                    
                with open(pre_dict_path, 'w', encoding='utf-8') as f:
                    json.dump({'preDict': pre_dict_serialized}, f, ensure_ascii=False, indent=2)
                    
                with open(pre_word_sum_path, 'w', encoding='utf-8') as f:
                    json.dump({'preWordSum': pre_word_sum_dict}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存前后继词关系时出错: {str(e)}")
                # 保存空的字典作为后备
                next_dict_path = os.path.join(preprocess_dir, "nextDict.json")
                next_word_sum_path = os.path.join(preprocess_dir, "nextWordSum.json")
                pre_dict_path = os.path.join(preprocess_dir, "preDict.json")
                pre_word_sum_path = os.path.join(preprocess_dir, "preWordSum.json")
                
                with open(next_dict_path, 'w', encoding='utf-8') as f:
                    json.dump({'nextDict': {}}, f, ensure_ascii=False, indent=2)
                    
                with open(next_word_sum_path, 'w', encoding='utf-8') as f:
                    json.dump({'nextWordSum': {}}, f, ensure_ascii=False, indent=2)
                    
                with open(pre_dict_path, 'w', encoding='utf-8') as f:
                    json.dump({'preDict': {}}, f, ensure_ascii=False, indent=2)
                    
                with open(pre_word_sum_path, 'w', encoding='utf-8') as f:
                    json.dump({'preWordSum': {}}, f, ensure_ascii=False, indent=2)
                
            # 保存首尾句
            try:
                sentence_path = os.path.join(preprocess_dir, "first_last_sentence.txt")
                with open(sentence_path, 'w', encoding='utf-8') as f:
                    f.write(f"{firstSentence}\n{lastSentence}")
            except Exception as e:
                print(f"保存首尾句时出错: {str(e)}")
        except Exception as e:
            print(f"保存预处理结果时出错: {str(e)}")
            # 出错时尝试保存最小有效数据
            try:
                empty_data_path = os.path.join(preprocess_dir, "empty_data.txt")
                with open(empty_data_path, 'w', encoding='utf-8') as f:
                    f.write("预处理出错，请重新运行预处理步骤。\n")
            except:
                pass

    def _load_preprocess_results(self, output_dir):
        """加载预处理结果"""
        try:
            # 规范化路径
            preprocess_dir = os.path.normpath(os.path.join(output_dir, "预处理"))
            
            # 确保目录存在
            if not os.path.exists(preprocess_dir):
                print(f"预处理目录不存在: {preprocess_dir}")
                return [], defaultdict(str), "", "", {}, defaultdict(int), {}, defaultdict(int)
            
            # 从文件加载wordsData
            words_data_path = os.path.join(preprocess_dir, "wordsData.txt")
            if not os.path.exists(words_data_path):
                print(f"wordsData文件不存在: {words_data_path}")
                return [], defaultdict(str), "", "", {}, defaultdict(int), {}, defaultdict(int)
                
            with open(words_data_path, 'r', encoding='utf-8') as f:
                wordsData = [line.strip() for line in f.readlines()]

            # 从文件加载wordsFlagDict
            words_flag_json_path = os.path.join(preprocess_dir, "wordsFlagDict.json")
            if not os.path.exists(words_flag_json_path):
                print(f"wordsFlagDict文件不存在: {words_flag_json_path}")
                wordsFlagDict = defaultdict(str)
            else:
                wordsFlagDict = outPut.loadDictFromJson(words_flag_json_path)

            # 加载其他必要的数据
            sentence_path = os.path.join(preprocess_dir, "first_last_sentence.txt")
            try:
                if not os.path.exists(sentence_path):
                    print(f"首尾句文件不存在: {sentence_path}")
                    firstSentence, lastSentence = "", ""
                else:
                    with open(sentence_path, 'r', encoding='utf-8') as f:
                        lines = f.read().strip().split('\n')
                        if len(lines) >= 2:
                            firstSentence, lastSentence = lines[0], lines[1]
                        else:
                            firstSentence, lastSentence = "", ""
            except FileNotFoundError:
                firstSentence, lastSentence = "", ""

            # 尝试加载前后继词字典，如果不存在则创建空字典
            next_dict_path = os.path.join(preprocess_dir, "nextDict.json")
            next_word_sum_path = os.path.join(preprocess_dir, "nextWordSum.json")
            pre_dict_path = os.path.join(preprocess_dir, "preDict.json")
            pre_word_sum_path = os.path.join(preprocess_dir, "preWordSum.json")
            
            try:
                if not os.path.exists(next_dict_path):
                    print(f"nextDict文件不存在: {next_dict_path}")
                    nextDict = {}
                else:
                    nextDict = outPut.loadDictFromJson(next_dict_path)
            except Exception as e:
                print(f"加载nextDict时出错: {str(e)}")
                nextDict = {}
                
            try:    
                if not os.path.exists(next_word_sum_path):
                    print(f"nextWordSum文件不存在: {next_word_sum_path}")
                    nextWordSum = defaultdict(int)
                else:
                    nextWordSum = outPut.loadDictFromJson(next_word_sum_path)
            except Exception as e:
                print(f"加载nextWordSum时出错: {str(e)}")
                nextWordSum = defaultdict(int)
                
            try:
                if not os.path.exists(pre_dict_path):
                    print(f"preDict文件不存在: {pre_dict_path}")
                    preDict = {}
                else:
                    preDict = outPut.loadDictFromJson(pre_dict_path)
            except Exception as e:
                print(f"加载preDict时出错: {str(e)}")
                preDict = {}
                
            try:
                if not os.path.exists(pre_word_sum_path):
                    print(f"preWordSum文件不存在: {pre_word_sum_path}")
                    preWordSum = defaultdict(int)
                else:
                    preWordSum = outPut.loadDictFromJson(pre_word_sum_path)
            except Exception as e:
                print(f"加载preWordSum时出错: {str(e)}")
                preWordSum = defaultdict(int)

            return wordsData, wordsFlagDict, firstSentence, lastSentence, nextDict, nextWordSum, preDict, preWordSum
        except Exception as e:
            print(f"加载预处理结果时出错: {str(e)}")
            # 返回空数据结构
            return [], defaultdict(str), "", "", {}, defaultdict(int), {}, defaultdict(int)

    def _save_semantic_features(self, interDensity, output_dir):
        """保存语义特征"""
        semantic_dir = os.path.normpath(os.path.join(output_dir, "语义特征"))
        os.makedirs(semantic_dir, exist_ok=True)
        
        txt_path = os.path.join(semantic_dir, "interDensity.txt")
        json_path = os.path.join(semantic_dir, "interDensity.json")
        
        outPut.writeDict(txt_path, interDensity)
        outPut.writeDictToJson(json_path, interDensity, 'interDensity')

    def _load_semantic_features(self, output_dir):
        """加载语义特征"""
        semantic_dir = os.path.normpath(os.path.join(output_dir, "语义特征"))
        json_path = os.path.join(semantic_dir, "interDensity.json")
        
        if not os.path.exists(json_path):
            print(f"语义特征文件不存在: {json_path}")
            return {}
            
        return outPut.loadDictFromJson(json_path)

    def _save_statistical_features(self, wordsTfidf, wordsLoc, wordsFlagWeight, output_dir):
        """保存统计特征"""
        statistical_dir = os.path.normpath(os.path.join(output_dir, "统计特征"))
        os.makedirs(statistical_dir, exist_ok=True)
        
        # 保存TF-IDF
        tfidf_txt_path = os.path.join(statistical_dir, "wordsTfidf.txt")
        tfidf_json_path = os.path.join(statistical_dir, "wordsTfidf.json")
        outPut.writeDict(tfidf_txt_path, wordsTfidf)
        outPut.writeDictToJson(tfidf_json_path, wordsTfidf, 'wordsTfidf')

        # 保存位置特征
        loc_txt_path = os.path.join(statistical_dir, "wordsLoc.txt")
        loc_json_path = os.path.join(statistical_dir, "wordsLoc.json")
        outPut.writeDict(loc_txt_path, wordsLoc)
        outPut.writeDictToJson(loc_json_path, wordsLoc, 'wordsLoc')

        # 保存词性权重
        flag_txt_path = os.path.join(statistical_dir, "wordsFlagWeight.txt")
        flag_json_path = os.path.join(statistical_dir, "wordsFlagWeight.json")
        outPut.writeDict(flag_txt_path, wordsFlagWeight)
        outPut.writeDictToJson(flag_json_path, wordsFlagWeight, 'wordsFlagWeight')

    def _load_statistical_features(self, output_dir):
        """加载统计特征"""
        try:
            # 创建统计特征目录路径
            statistical_dir = os.path.normpath(os.path.join(output_dir, "统计特征"))
            
            # 加载TF-IDF特征
            wordsTfidf_path = os.path.join(statistical_dir, "wordsTfidf.json")
            try:
                if not os.path.exists(wordsTfidf_path):
                    print(f"TF-IDF特征文件不存在: {wordsTfidf_path}")
                    wordsTfidf = {}
                else:
                    wordsTfidf = outPut.loadDictFromJson(wordsTfidf_path)
                    print(f"成功加载TF-IDF特征，共有 {len(wordsTfidf)} 个词")
            except Exception as e:
                print(f"加载TF-IDF特征时出错: {str(e)}")
                wordsTfidf = {}
            
            # 加载位置特征
            wordsLoc_path = os.path.join(statistical_dir, "wordsLoc.json")
            try:
                if not os.path.exists(wordsLoc_path):
                    print(f"位置特征文件不存在: {wordsLoc_path}")
                    wordsLoc = {}
                else:
                    wordsLoc = outPut.loadDictFromJson(wordsLoc_path)
                    print(f"成功加载位置特征，共有 {len(wordsLoc)} 个词")
            except Exception as e:
                print(f"加载位置特征时出错: {str(e)}")
                wordsLoc = {}
            
            # 加载词性特征
            wordsFlagWeight_path = os.path.join(statistical_dir, "wordsFlagWeight.json")
            try:
                if not os.path.exists(wordsFlagWeight_path):
                    print(f"词性特征文件不存在: {wordsFlagWeight_path}")
                    wordsFlagWeight = {}
                else:
                    wordsFlagWeight = outPut.loadDictFromJson(wordsFlagWeight_path)
                    print(f"成功加载词性特征，共有 {len(wordsFlagWeight)} 个词")
            except Exception as e:
                print(f"加载词性特征时出错: {str(e)}")
                wordsFlagWeight = {}
            
            # 加载长度特征（可选）
            wordsLen_path = os.path.join(statistical_dir, "wordsLen.json")
            try:
                if not os.path.exists(wordsLen_path):
                    print(f"长度特征文件不存在: {wordsLen_path}")
                    wordsLen = {}
                else:
                    wordsLen = outPut.loadDictFromJson(wordsLen_path)
                    print(f"成功加载长度特征，共有 {len(wordsLen)} 个词")
            except Exception as e:
                print(f"加载长度特征时出错: {str(e)}")
                wordsLen = {}
            
            return wordsTfidf, wordsLoc, wordsFlagWeight, wordsLen
        except Exception as e:
            print(f"加载统计特征时出错: {str(e)}")
            return {}, {}, {}, {}

    def _save_scores(self, score, output_dir):
        """保存词语得分"""
        try:
            # 创建保存目录
            score_dir = os.path.normpath(os.path.join(output_dir, "词语得分"))
            os.makedirs(score_dir, exist_ok=True)
            
            # 创建保存路径
            score_txt_path = os.path.join(score_dir, "score.txt")
            score_json_path = os.path.join(score_dir, "score.json")
            
            # 确保score是字典类型
            if isinstance(score, list):
                # 如果是排序后的列表，转换回字典
                score_dict = {word: score_val for word, score_val in score}
            elif isinstance(score, dict):
                score_dict = score
            else:
                print(f"警告: 得分不是有效的类型: {type(score)}")
                if hasattr(score, 'items'):
                    score_dict = dict(score)
                else:
                    print("错误: 无法将得分转换为字典")
                    return False
            
            # 保存到文本和JSON文件
            outPut.writeDict(score_txt_path, score_dict)
            outPut.writeDictToJson(score_json_path, score_dict, 'score')
            
            print(f"词语得分已保存到: {score_txt_path}")
            return True
        except Exception as e:
            print(f"保存词语得分时出错: {str(e)}")
            return False

    def _save_feature(self, feature_dict, feature_name, output_dir):
        """保存特征到文本和JSON文件"""
        try:
            # 确保特征是字典类型
            if not isinstance(feature_dict, dict):
                print(f"警告: {feature_name}不是字典类型: {type(feature_dict)}")
                if hasattr(feature_dict, 'items'):
                    feature_dict = dict(feature_dict)
                else:
                    print(f"错误: 无法将{feature_name}转换为字典，保存失败")
                    return False
            
            # 创建文件路径
            txt_path = os.path.join(output_dir, f"{feature_name}.txt")
            json_path = os.path.join(output_dir, f"{feature_name}.json")
            
            # 保存到文本和JSON文件
            outPut.writeDict(txt_path, feature_dict)
            outPut.writeDictToJson(json_path, feature_dict, feature_name)
            
            return True
        except Exception as e:
            print(f"保存特征{feature_name}时出错: {str(e)}")
            return False

    def _save_keywords(self, keywords, output_dir):
        """保存提取的关键词"""
        try:
            # 创建保存目录
            keyword_dir = os.path.normpath(os.path.join(output_dir, "词语得分"))
            os.makedirs(keyword_dir, exist_ok=True)
            
            # 创建保存路径
            keyword_txt_path = os.path.join(keyword_dir, "keywords.txt")
            keyword_json_path = os.path.join(keyword_dir, "keywords.json")
            
            # 格式化关键词
            formatted_keywords = []
            for i, (word, score) in enumerate(keywords, 1):
                formatted_keywords.append(f"{i}. {word} ({score:.4f})")
            
            # 保存到文本文件
            with open(keyword_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(formatted_keywords))
            
            # 保存到JSON文件
            keyword_dict = {word: score for word, score in keywords}
            outPut.writeDictToJson(keyword_json_path, keyword_dict, 'keywords')
            
            print(f"关键词已保存到: {keyword_txt_path}")
            return True
        except Exception as e:
            print(f"保存关键词时出错: {str(e)}")
            return False 