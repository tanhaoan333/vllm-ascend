import asyncio
import aiohttp
import time
import json
import os
import re
from statistics import mean
from pathlib import Path
from typing import Optional
import difflib

# ==================== 配置参数 ====================
# ASR 服务配置
ASR_URL = "http://localhost:8055/v1/chat/completions"
MAX_CONCURRENT = 5  # 最大并发数
MAX_TOKENS = 100  # 最大生成token数

# AiShell-1 数据集路径配置
# 请根据实际路径修改
AISHELL_DATA_ROOT = "/home/speech_asr_aishell1_testsets"  # 数据集根目录
WAV_DIR = "/home/speech_asr_aishell1_testsets/wav/test"  # 音频文件目录
TRANSCRIPT_FILE = "/home/speech_asr_aishell1_testsets/transcript/data.text" # 标注文件路径

# 测试结果保存路径
RESULTS_FILE = "aishell_results.json"

# ==================== 全局变量 ====================
all_results = []


# ==================== 辅助函数 ====================
def normalize_chinese_text(text: str) -> str:
    """标准化中文文本：去除标点、空格，统一格式"""
    # 去除标点符号
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    # 去除空格
    text = text.replace(' ', '').strip()
    # 统一转换为小写（如果有英文）
    return text.lower()


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    计算字符错误率 (Character Error Rate)
    使用编辑距离算法
    """
    ref = normalize_chinese_text(reference)
    hyp = normalize_chinese_text(hypothesis)
    
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    
    # 动态规划计算编辑距离
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
    
    cer = dp[m][n] / m
    return cer


def load_aishell_transcript(transcript_path: str) -> dict:
    """
    加载 AiShell-1 标注文件
    格式: 音频文件名 文本内容
    例如: BAC009S0002W0122 把音量切成四十
    """
    transcripts = {}
    if not os.path.exists(transcript_path):
        print(f"警告: 标注文件不存在: {transcript_path}")
        return transcripts
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                audio_id, text = parts
                transcripts[audio_id] = text
            elif len(parts) == 1:
                transcripts[parts[0]] = ""
    
    return transcripts


def find_audio_file(audio_id: str, wav_root: str) -> Optional[str]:
    """
    根据音频ID查找音频文件路径
    AiShell-1 音频文件结构: wav/{speaker_id}/{audio_id}.wav
    例如: wav/S0002/BAC009S0002W0122.wav
    """
    speaker_id = audio_id[7:12]  # 从 BAC009S0002W0122 提取 S0002
    
    # 可能的文件路径
    possible_paths = [
        os.path.join(wav_root, speaker_id, f"{audio_id}.wav"),
        os.path.join(wav_root, f"{audio_id}.wav"),
        os.path.join(wav_root, speaker_id, audio_id.replace('.wav', '') + '.wav'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def file_to_data_url(file_path: str) -> str:
    """将本地音频文件转换为 data URL"""
    import base64
    
    with open(file_path, 'rb') as f:
        audio_data = f.read()
    
    # WAV文件使用 audio/wav MIME类型
    encoded = base64.b64encode(audio_data).decode('utf-8')
    return f"data:audio/wav;base64,{encoded}"


# ==================== ASR 请求函数 ====================
async def make_asr_request(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    audio_path: str,
    audio_id: str,
    reference_text: str
) -> dict:
    """
    发送 ASR 请求并计算性能指标
    """
    async with semaphore:
        # 准备音频数据
        try:
            audio_url = file_to_data_url(audio_path)
        except Exception as e:
            return {
                'audio_id': audio_id,
                'success': False,
                'error': f'读取音频文件失败: {e}',
                'reference': reference_text,
                'hypothesis': '',
                'cer': None,
                'ttft': None,
                'tpot': None,
                'e2e_time': None,
                'tokens': 0
            }
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}}
                    ]
                }
            ],
            "stream": True,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0  # ASR任务使用贪婪解码
        }
        
        request_start = time.time()
        ttft = None
        tpot_list = []
        token_count = 0
        recognized_text = ""
        
        try:
            async with session.post(ASR_URL, json=payload) as resp:
                first_token_time = None
                last_token_time = None
                
                async for line in resp.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                current_time = time.time()
                                
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    
                                    if 'content' in delta:
                                        content = delta['content']
                                        if content:
                                            token_count += 1
                                            recognized_text += content
                                            
                                            if first_token_time is None:
                                                first_token_time = current_time
                                                ttft = first_token_time - request_start
                                            else:
                                                if last_token_time is not None:
                                                    tpot = current_time - last_token_time
                                                    tpot_list.append(tpot)
                                            
                                            last_token_time = current_time
                            
                            except json.JSONDecodeError:
                                continue
                
                e2e_time = time.time() - request_start if last_token_time else None
                
        except Exception as e:
            return {
                'audio_id': audio_id,
                'success': False,
                'error': f'请求失败: {e}',
                'reference': reference_text,
                'hypothesis': '',
                'cer': None,
                'ttft': None,
                'tpot': None,
                'e2e_time': None,
                'tokens': 0
            }
        recognized_text = extract_clean_text(recognized_text)
        # 计算 CER
        cer = compute_cer(reference_text, recognized_text)
        
        return {
            'audio_id': audio_id,
            'success': True,
            'error': None,
            'reference': reference_text,
            'hypothesis': recognized_text,
            'cer': cer,
            'ttft': ttft,
            'tpot': mean(tpot_list) if tpot_list else None,
            'e2e_time': e2e_time,
            'tokens': token_count
        }


# ==================== 主测试函数 ====================
async def run_aishell_test(
    session: aiohttp.ClientSession,
    max_concurrent: int,
    test_samples: Optional[int] = None
):
    """
    运行 AiShell-1 测试
    
    Args:
        test_samples: 如果指定，则只测试前 N 个样本（用于快速测试）
    """
    print("=" * 80)
    print("AiShell-1 语音识别测试")
    print("=" * 80)
    
    # 加载标注
    print(f"\n加载标注文件: {TRANSCRIPT_FILE}")
    transcripts = load_aishell_transcript(TRANSCRIPT_FILE)
    print(f"共加载 {len(transcripts)} 条标注")
    
    if len(transcripts) == 0:
        print("错误: 未加载到任何标注，请检查路径配置")
        return
    
    # 准备测试列表
    test_items = []
    for audio_id, reference in transcripts.items():
        audio_path = find_audio_file(audio_id, WAV_DIR)
        if audio_path:
            test_items.append((audio_id, audio_path, reference))
        else:
            print(f"警告: 未找到音频文件: {audio_id}")
    
    print(f"找到 {len(test_items)} 个有效音频文件")
    
    # 限制测试样本数
    if test_samples and test_samples < len(test_items):
        test_items = test_items[:test_samples]
        print(f"将测试前 {test_samples} 个样本")
    
    # 创建并发控制
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 创建任务
    print(f"\n开始测试 (并发数: {max_concurrent})...")
    tasks = [
        asyncio.create_task(
            make_asr_request(session, semaphore, audio_path, audio_id, reference)
        )
        for audio_id, audio_path, reference in test_items
    ]
    
    # 执行测试
    results = await asyncio.gather(*tasks)
    
    return results

def extract_clean_text(raw_text: str) -> str:
    """从包含 <asr_text> 标签的响应中抽取纯文本"""
    match = re.search(r'<asr_text>(.*?)(?:</asr_text>|$)', raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def analyze_results(results: list):
    """分析测试结果"""
    print("\n" + "=" * 80)
    print("测试结果分析")
    print("=" * 80)
    
    # 统计成功/失败
    success_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n总样本数: {len(results)}")
    print(f"成功: {len(success_results)}")
    print(f"失败: {len(failed_results)}")
    
    if failed_results:
        print("\n失败样本:")
        for r in failed_results[:5]:  # 只显示前5个失败
            print(f"  - {r['audio_id']}: {r['error']}")
    
    if not success_results:
        print("\n没有成功的测试结果")
        return
    
    # CER 统计
    cers = [r['cer'] for r in success_results if r['cer'] is not None]
    if cers:
        avg_cer = mean(cers)
        min_cer = min(cers)
        max_cer = max(cers)
        
        print(f"\n字符错误率 (CER):")
        print(f"  平均 CER: {avg_cer:.2%} ({avg_cer:.4f})")
        print(f"  最小 CER: {min_cer:.2%} ({min_cer:.4f})")
        print(f"  最大 CER: {max_cer:.2%} ({max_cer:.4f})")
        
        # 准确率
        accuracy = 1 - avg_cer
        print(f"  准确率: {accuracy:.2%}")
    
    # 性能指标
    ttfts = [r['ttft'] for r in success_results if r['ttft'] is not None]
    tpots = [r['tpot'] for r in success_results if r['tpot'] is not None]
    e2e_times = [r['e2e_time'] for r in success_results if r['e2e_time'] is not None]
    
    print(f"\n性能指标:")
    if ttfts:
        print(f"  TTFT (首字符延迟):")
        print(f"    平均: {mean(ttfts) * 1000:.1f}ms")
        print(f"    最小: {min(ttfts) * 1000:.1f}ms")
        print(f"    最大: {max(ttfts) * 1000:.1f}ms")
    
    if tpots:
        print(f"  TPOT (字符间延迟):")
        print(f"    平均: {mean(tpots) * 1000:.1f}ms")
        print(f"    最小: {min(tpots) * 1000:.1f}ms")
        print(f"    最大: {max(tpots) * 1000:.1f}ms")
    
    if e2e_times:
        print(f"  端到端延迟:")
        print(f"    平均: {mean(e2e_times) * 1000:.1f}ms")
        print(f"    最小: {min(e2e_times) * 1000:.1f}ms")
        print(f"    最大: {max(e2e_times) * 1000:.1f}ms")
    
    # 显示一些识别示例
    print(f"\n识别示例 (前 10 条):")
    for r in success_results[:10]:
        cer_str = f"CER: {r['cer']:.2%}" if r['cer'] is not None else "CER: N/A"
        print(f"  {r['audio_id']} [{cer_str}]")
        print(f"    参考: {r['reference']}")
        print(f"    识别: {r['hypothesis']}")
        print()
    
    # 显示 CER 最高的样本（错误最多的）
    if cers:
        print(f"\n错误率最高的样本 (Top 5):")
        sorted_by_cer = sorted(success_results, key=lambda x: x['cer'] or 0, reverse=True)
        for r in sorted_by_cer[:5]:
            print(f"  {r['audio_id']} [CER: {r['cer']:.2%}]")
            print(f"    参考: {r['reference']}")
            print(f"    识别: {r['hypothesis']}")
            print()
    
    return {
        'total': len(results),
        'success': len(success_results),
        'failed': len(failed_results),
        'avg_cer': mean(cers) if cers else None,
        'ttft_avg_ms': mean(ttfts) * 1000 if ttfts else None,
        'tpot_avg_ms': mean(tpots) * 1000 if tpots else None,
        'e2e_avg_ms': mean(e2e_times) * 1000 if e2e_times else None,
    }


def save_results(results: list, summary: dict):
    """保存结果到文件"""
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'asr_url': ASR_URL,
            'max_concurrent': MAX_CONCURRENT,
            'max_tokens': MAX_TOKENS
        },
        'summary': summary,
        'results': results
    }
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {RESULTS_FILE}")


# ==================== 主函数 ====================
async def main():
    # 检查配置
    print("配置检查:")
    print(f"  ASR URL: {ASR_URL}")
    print(f"  数据集路径: {AISHELL_DATA_ROOT}")
    print(f"  标注文件: {TRANSCRIPT_FILE}")
    print(f"  音频目录: {WAV_DIR}")
    
    if AISHELL_DATA_ROOT == "/path/to/aishell1":
        print("\n警告: 请先修改脚本中的 AISHELL_DATA_ROOT 为实际的数据集路径")
        print("例如: AISHELL_DATA_ROOT = '/data/aishell1' or AISHELL_DATA_ROOT = '/home/user/data/aishell1'")
        return
    
    async with aiohttp.ClientSession() as session:
        # 运行测试（可以指定测试样本数，如 test_samples=10 快速测试）
        results = await run_aishell_test(session, MAX_CONCURRENT, test_samples=None)
        
        # 分析结果
        summary = analyze_results(results)
        
        # 保存结果
        save_results(results, summary)


if __name__ == "__main__":
    asyncio.run(main())
