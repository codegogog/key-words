import json
import os

def writeDict(filePath, dict_data):
    """将字典写入文本文件"""
    try:
        with open(filePath, 'w', encoding='utf-8') as file_object:
            if not isinstance(dict_data, dict):
                print(f"警告: 输入不是字典类型: {type(dict_data)}")
                # 尝试转换为字典
                if hasattr(dict_data, 'items'):
                    dict_data = dict(dict_data)
                else:
                    print(f"错误: 无法将输入转换为字典")
                    return False
                    
            for key, value in dict_data.items():
                if isinstance(value, dict):
                    file_object.write(f"{key}\t{json.dumps(value, ensure_ascii=False)}\n")
                else:
                    file_object.write(f"{key}\t{value}\n")
        return True
    except Exception as e:
        print(f"写入字典到文本文件时出错: {str(e)}")
        return False

def writeDictToJson(filePath, dict_data, dict_name):
    """将字典写入JSON文件"""
    try:
        with open(filePath, 'w', encoding='utf-8') as file_object:
            if not isinstance(dict_data, dict):
                print(f"警告: 输入不是字典类型: {type(dict_data)}")
                # 尝试转换为字典
                if hasattr(dict_data, 'items'):
                    dict_data = dict(dict_data)
                else:
                    print(f"错误: 无法将输入转换为字典")
                    # 写入空字典
                    json.dump({dict_name: {}}, file_object, ensure_ascii=False, indent=2)
                    return False
                    
            json.dump({dict_name: dict_data}, file_object, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"写入字典到JSON文件时出错: {str(e)}")
        # 尝试写入空字典作为后备
        try:
            with open(filePath, 'w', encoding='utf-8') as file_object:
                json.dump({dict_name: {}}, file_object, ensure_ascii=False, indent=2)
        except:
            pass
        return False

def loadDictFromJson(filePath):
    """从JSON文件加载字典"""
    try:
        # 标准化路径
        filePath = os.path.normpath(filePath)
        
        if not os.path.exists(filePath):
            print(f"JSON文件不存在: {filePath}")
            return {}
            
        with open(filePath, 'r', encoding='utf-8') as file_object:
            try:
                data = json.load(file_object)
                # 获取第一个键的值（字典名称）
                if data and isinstance(data, dict):
                    result = next(iter(data.values()), {})
                    # 确保返回的是字典
                    if not isinstance(result, dict):
                        print(f"警告: 从JSON加载的数据不是字典类型: {type(result)}")
                        if hasattr(result, 'items'):
                            result = dict(result)
                        else:
                            print(f"错误: 无法转换为字典，返回空字典")
                            return {}
                    return result
                return {}
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}, 文件路径: {filePath}")
                return {}
    except Exception as e:
        print(f"加载JSON文件时发生错误: {str(e)}, 文件路径: {filePath}")
        return {}

def writeToTxt(filePath, list_data):
    """将列表写入文本文件"""
    try:
        with open(filePath, 'w', encoding='utf-8') as file_object:
            # 确保输入是列表类型
            if not isinstance(list_data, (list, tuple)):
                print(f"警告: 输入不是列表类型: {type(list_data)}")
                # 尝试转换为列表
                if hasattr(list_data, '__iter__') and not isinstance(list_data, (str, dict)):
                    list_data = list(list_data)
                else:
                    print(f"错误: 无法将输入转换为列表")
                    # 写入空列表
                    file_object.write("")
                    return False
                    
            for item in list_data:
                if isinstance(item, tuple) and len(item) >= 2:
                    file_object.write(f"{item[0]}\t{item[1]}\n")
                else:
                    file_object.write(f"{item}\n")
        return True
    except Exception as e:
        print(f"写入列表到文本文件时出错: {str(e)}")
        return False

def writeToJson(filePath, list_data):
    """将列表写入JSON文件"""
    try:
        with open(filePath, 'w', encoding='utf-8') as file_object:
            # 确保输入是列表类型
            if not isinstance(list_data, (list, tuple)):
                print(f"警告: 输入不是列表类型: {type(list_data)}")
                # 尝试转换为列表
                if hasattr(list_data, '__iter__') and not isinstance(list_data, (str, dict)):
                    list_data = list(list_data)
                else:
                    print(f"错误: 无法将输入转换为列表")
                    # 写入空列表
                    json.dump([], file_object, ensure_ascii=False, indent=2)
                    return False
                    
            json.dump(list_data, file_object, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"写入列表到JSON文件时出错: {str(e)}")
        # 尝试写入空列表作为后备
        try:
            with open(filePath, 'w', encoding='utf-8') as file_object:
                json.dump([], file_object, ensure_ascii=False, indent=2)
        except:
            pass
        return False

def loadFromJson(filePath):
    """从JSON文件加载列表"""
    try:
        # 标准化路径
        filePath = os.path.normpath(filePath)
        
        if not os.path.exists(filePath):
            print(f"JSON文件不存在: {filePath}")
            return []
            
        with open(filePath, 'r', encoding='utf-8') as file_object:
            try:
                data = json.load(file_object)
                # 确保返回的是列表类型
                if not isinstance(data, list):
                    print(f"警告: 从JSON加载的数据不是列表类型: {type(data)}")
                    # 尝试转换为列表
                    if hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
                        return list(data)
                    else:
                        print(f"错误: 无法转换为列表，返回空列表")
                        return []
                return data
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}, 文件路径: {filePath}")
                return []
    except Exception as e:
        print(f"加载JSON文件时发生错误: {str(e)}, 文件路径: {filePath}")
        return []