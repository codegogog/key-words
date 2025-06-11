import os
import re
import jieba
import jieba.posseg as pseg
from collections import defaultdict

# 允许的词性列表
ALLOW_SPEECH_TAGS = ('a', 'ad', 'an', 'i', 'j', 'l', 'v', 'vg', 'vd', 'vn', 'n', 'ns', 'nsf', 'nt', 'nz')
NOT_ALLOW_TAGS = ('x', 'w')

def initialize_jieba():
    """初始化jieba分词器"""
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 尝试加载主词典
    dict_path = os.path.join(current_dir, 'dict_file', 'dict.txt.big')
    if os.path.exists(dict_path):
        try:
            jieba.set_dictionary(dict_path)
            print(f"已加载主词典: {dict_path}")
        except Exception as e:
            print(f"加载主词典失败: {str(e)}")

    # 尝试加载用户词典
    user_dict_path = os.path.join(current_dir, 'dict_file', 'user_dict.txt')
    if os.path.exists(user_dict_path):
        try:
            jieba.load_userdict(user_dict_path)
            print(f"已加载用户词典: {user_dict_path}")
        except Exception as e:
            print(f"加载用户词典失败: {str(e)}")

    # 初始化jieba
    jieba.initialize()
    return True

def load_stop_words():
    """加载停用词"""
    stop_words = []
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stop_words_path = os.path.join(current_dir, 'dict_file', 'stop_words.txt')
        if os.path.exists(stop_words_path):
            with open(stop_words_path, 'r', encoding='utf-8') as f:
                stop_words = [line.strip() for line in f.readlines()]
            print(f"已加载 {len(stop_words)} 个停用词")
        else:
            print(f"停用词文件不存在: {stop_words_path}")
    except Exception as e:
        print(f"加载停用词时出错: {str(e)}")
    return stop_words

def split_sentences(text):
    """按标点符号分句"""
    pattern = r'。|！|？|;'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def safe_cut(text, use_paddle=False):
    """安全地使用jieba进行分词，防止可能的错误"""
    try:
        if not text or not isinstance(text, str):
            print(f"警告: 分词输入无效或不是字符串: {type(text)}")
            return []
        
        print(f"开始分词，文本长度: {len(text)}字符")
        if len(text) > 1000:
            # 如果文本过长，输出前50个字符作为示例
            print(f"分词文本示例: {text[:50]}...")
        else:
            print(f"分词文本: {text}")
        
        # 尝试不同的分词模式
        if use_paddle:
            try:
                print("使用paddle模式进行分词...")
                words_with_pos = pseg.cut(text, use_paddle=True)
            except Exception as e:
                print(f"使用paddle模式分词失败: {str(e)}，切换为普通模式")
                print("使用普通模式进行分词...")
                words_with_pos = pseg.cut(text)
        else:
            print("使用普通模式进行分词...")
            words_with_pos = pseg.cut(text)
            
        # 将生成器转换为列表，以便多次使用
        result = list(words_with_pos)
        print(f"分词完成，得到 {len(result)} 个词语")
        
        # 输出分词结果示例
        if result:
            if len(result) > 10:
                sample = result[:10]
                print(f"分词结果示例: {[(word.word, word.flag) for word in sample]}...")
            else:
                print(f"分词结果: {[(word.word, word.flag) for word in result]}")
        else:
            print("警告: 分词结果为空")
            
        return result
    except ImportError as ie:
        print(f"导入分词模块失败: {str(ie)}")
        return []
    except Exception as e:
        print(f"分词过程中出错: {str(e)}")
        # 输出完整错误堆栈
        import traceback
        traceback.print_exc()
        # 出错时返回空列表
        return []

def extract_candidate_words(text, title, stop_words):
    """提取候选关键词"""
    try:
        words_data = []  # 候选关键词列表
        words_flag_dict = {}  # 词语-词性映射
        
        # 确保停用词是列表或集合
        if not isinstance(stop_words, (list, set, tuple)):
            print(f"警告: 停用词不是集合类型: {type(stop_words)}")
            if hasattr(stop_words, '__iter__') and not isinstance(stop_words, str):
                stop_words = list(stop_words)
            else:
                stop_words = []
        
        # 处理正文
        if text and isinstance(text, str):
            words_with_pos = safe_cut(text)
            for word_pos in words_with_pos:
                if (hasattr(word_pos, 'word') and hasattr(word_pos, 'flag') and 
                    len(word_pos.word) > 1 and 
                    word_pos.flag in ALLOW_SPEECH_TAGS and 
                    word_pos.word not in stop_words):
                    words_data.append(word_pos.word)
                    words_flag_dict[word_pos.word] = word_pos.flag
        
        # 处理标题（标题词语权重更高）
        if title and isinstance(title, str):
            title_words = safe_cut(title)
            for word_pos in title_words:
                if (hasattr(word_pos, 'word') and hasattr(word_pos, 'flag') and 
                    len(word_pos.word) > 1 and 
                    word_pos.flag in ALLOW_SPEECH_TAGS and 
                    word_pos.word not in stop_words):
                    words_data.append(word_pos.word)
                    words_flag_dict[word_pos.word] = word_pos.flag
        
        # 去重
        words_data = list(set(words_data))
        
        return words_data, words_flag_dict
    except Exception as e:
        print(f"提取候选关键词时出错: {str(e)}")
        return [], {}

def simple_preprocess(text, title):
    """简化的预处理过程"""
    try:
        print("开始简化预处理...")
        
        # 验证输入类型
        if not isinstance(text, str):
            print(f"警告: 输入文本不是字符串类型: {type(text)}")
            text = str(text) if text is not None else ""
            
        if not isinstance(title, str):
            print(f"警告: 输入标题不是字符串类型: {type(title)}")
            title = str(title) if title is not None else ""
        
        # 1. 初始化jieba
        initialize_jieba()
        
        # 2. 加载停用词
        stop_words = load_stop_words()
        
        # 3. 提取候选关键词
        words_data, words_flag_dict = extract_candidate_words(text, title, stop_words)
        print(f"预处理完成，共提取 {len(words_data)} 个候选关键词")
        
        # 4. 提取首尾句
        sentences = split_sentences(text)
        first_sentence = sentences[0] if sentences else ""
        last_sentence = sentences[-1] if sentences else ""
        
        # 5. 创建空的前后继词字典（简化版不做复杂处理）
        next_dict = {}  # 明确初始化为空字典
        next_word_sum = defaultdict(int)  # 使用默认值为0的字典
        pre_dict = {}  # 明确初始化为空字典
        pre_word_sum = defaultdict(int)  # 使用默认值为0的字典
        
        return words_data, words_flag_dict, first_sentence, last_sentence, next_dict, next_word_sum, pre_dict, pre_word_sum
        
    except Exception as e:
        print(f"简化预处理出错: {str(e)}")
        # 出错时返回最小可用数据结构
        return [], {}, "", "", {}, defaultdict(int), {}, defaultdict(int) 