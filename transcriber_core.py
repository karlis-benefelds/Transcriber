import gc
import whisper
import json
import datetime
from pathlib import Path
import torch
from pydub import AudioSegment
import numpy as np
import requests
import re
import iso8601
import csv
import subprocess
from datetime import timedelta
from tqdm import tqdm
import threading
import tempfile
from contextlib import nullcontext

class TranscriberService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        
    def start_transcription(self, job_id, curl_command, audio_source, privacy_mode, status_callback):
        """Start transcription in a background thread"""
        thread = threading.Thread(
            target=self._transcribe_worker,
            args=(job_id, curl_command, audio_source, privacy_mode, status_callback)
        )
        thread.daemon = True
        thread.start()
    
    def _transcribe_worker(self, job_id, curl_command, audio_source, privacy_mode, status_callback):
        try:
            # Extract class ID
            status_callback({'progress': 5, 'message': 'Extracting class information...'})
            class_id = self._extract_class_id(curl_command)
            
            # Fetch Forum data
            status_callback({'progress': 10, 'message': 'Fetching class data from Forum...'})
            headers = self._clean_curl(curl_command)
            events_data = self._get_forum_events(class_id, headers)
            
            # Process audio
            status_callback({'progress': 20, 'message': 'Processing audio file...'})
            audio_processor = AudioPreprocessor()
            processed_audio_path = audio_processor.validate_and_fix_file(audio_source)
            
            # Transcribe
            status_callback({'progress': 30, 'message': 'Starting transcription...'})
            transcript_processor = TranscriptionProcessor(
                progress_callback=lambda p: status_callback({
                    'progress': 30 + int(p * 0.5), 
                    'message': f'Transcribing audio... {int(p)}%'
                })
            )
            transcript_path = transcript_processor.transcribe(processed_audio_path, class_id)
            
            # Generate outputs
            status_callback({'progress': 85, 'message': 'Generating PDF and CSV files...'})
            output_generator = OutputGenerator()
            
            if privacy_mode == 'both':
                # Generate both versions
                pdf_names, csv_names = output_generator.generate_outputs(
                    class_id, headers, events_data, transcript_path, 'names'
                )
                pdf_ids, csv_ids = output_generator.generate_outputs(
                    class_id, headers, events_data, transcript_path, 'ids'
                )
                
                # Complete with both file paths
                status_callback({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Transcription completed successfully!',
                    'privacy_mode': privacy_mode,
                    'class_id': class_id,
                    'class_name': events_data.get('class_meta', {}).get('session_title', class_id),
                    'pdf_path_names': pdf_names,
                    'csv_path_names': csv_names,
                    'pdf_path_ids': pdf_ids,
                    'csv_path_ids': csv_ids
                })
            else:
                # Generate single version
                pdf_path, csv_path = output_generator.generate_outputs(
                    class_id, headers, events_data, transcript_path, privacy_mode
                )
                
                # Complete
                status_callback({
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Transcription completed successfully!',
                    'privacy_mode': privacy_mode,
                    'class_id': class_id,
                    'class_name': events_data.get('class_meta', {}).get('session_title', class_id),
                    'pdf_path': pdf_path,
                    'csv_path': csv_path
                })
            
        except Exception as e:
            status_callback({
                'status': 'error',
                'message': f'Error: {str(e)}'
            })
    
    def _extract_class_id(self, curl_text):
        """Extract class ID from cURL command"""
        ids = self._extract_ids_from_curl(curl_text)
        class_id = ids.get("class_id") or (
            re.search(r"/api/v1/class_grader/classes/(\d+)", curl_text).group(1)
            if re.search(r"/api/v1/class_grader/classes/(\d+)", curl_text) else None
        )
        if not class_id:
            raise ValueError(
                "Could not extract Class ID from your cURL. "
                "Make sure the cURL includes a class URL."
            )
        return class_id
    
    def _extract_ids_from_curl(self, curl_text):
        """Pull class/section/course IDs from cURL"""
        ref_match = re.search(r"-H\s+['\"](?:referer|Referer):\s*([^'\"\r\n]+)", curl_text)
        ref = ref_match.group(1).strip() if ref_match else ""
        class_link = ""
        course_id = section_id = class_id = None

        if ref:
            m = re.search(r"/app/courses/(\d+)/sections/(\d+)/classes/(\d+)", ref)
            if m:
                course_id, section_id, class_id = m.group(1), m.group(2), m.group(3)
                class_link = ref

        if not class_id:
            m2 = re.search(r"/api/v1/class_grader/classes/(\d+)", curl_text)
            if m2:
                class_id = m2.group(1)
                class_link = f"https://forum.minerva.edu/app/classes/{class_id}"

        return {
            "course_id": course_id,
            "section_id": section_id,
            "class_id": class_id,
            "class_link": class_link
        }
    
    def _clean_curl(self, curl_string):
        """Parse cURL and return headers dict"""
        headers = {}
        header_matches = re.findall(r"-H ['\"](.*?): (.*?)['\"]", curl_string)
        for name, value in header_matches:
            headers[name] = value
        cookie_match = re.search(r"-b ['\"](.*?)['\"]", curl_string)
        if cookie_match:
            headers['Cookie'] = cookie_match.group(1)
        return headers
    
    def _get_forum_events(self, class_id, headers):
        """Fetch class metadata and events from Forum"""
        # Class meta
        class_url = f'https://forum.minerva.edu/api/v1/class_grader/classes/{class_id}'
        r = requests.get(class_url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to access class data. Status code: {r.status_code}")
        data = r.json()

        session_title = data.get('title') or f"Session {class_id}"
        course_obj = (data.get('section') or {}).get('course') or {}
        course_code = course_obj.get('course-code', '')
        course_title = course_obj.get('title', '')
        section_title = (data.get('section') or {}).get('title', '')
        class_type = data.get('type', '')
        rec = (data.get('recording-sessions') or [{}])[0]
        recording_start = rec.get('recording-started')
        recording_end = rec.get('recording-ended')

        schedule_guess = ''
        if isinstance(section_title, str) and ',' in section_title:
            parts = [p.strip() for p in section_title.split(',', 1)]
            schedule_guess = parts[1] if len(parts) > 1 else ''

        class_meta = {
            'session_title': session_title,
            'course_code': course_code,
            'course_title': course_title,
            'section_title': section_title,
            'schedule': schedule_guess,
            'class_type': class_type,
            'recording_start': recording_start,
            'recording_end': recording_end,
        }

        if not recording_start:
            raise KeyError("No recording-started found in class data")

        # Events
        events_url = f'https://forum.minerva.edu/api/v1/class_grader/classes/{class_id}/class-events'
        r = requests.get(events_url, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to access class events. Status code: {r.status_code}")
        events = r.json()
        if not isinstance(events, list):
            raise ValueError("No valid class events returned from API")

        voice_events = []
        timeline_segments = []
        ref_time = iso8601.parse_date(recording_start)

        for ev in events:
            et = ev.get('event-type')
            try:
                if et == 'voice':
                    duration_ms = (ev.get('event-data') or {}).get('duration', 0)
                    duration = duration_ms / 1000.0
                    if duration >= 1:
                        start_time = iso8601.parse_date(ev['start-time'])
                        end_time = iso8601.parse_date(ev['end-time'])
                        voice_events.append({
                            'start': (start_time - ref_time).total_seconds(),
                            'end': (end_time - ref_time).total_seconds(),
                            'duration': duration,
                            'speaker': {
                                'id': (ev.get('actor') or {}).get('id') or (ev.get('actor') or {}).get('user-id') or ((ev.get('actor') or {}).get('user') or {}).get('id'),
                                'first-name': (ev.get('actor') or {}).get('first-name'),
                                'last-name': (ev.get('actor') or {}).get('last-name')
                            }
                        })
                elif et == 'timeline-segment':
                    start_time = iso8601.parse_date(ev['start-time'])
                    seg = (ev.get('event-data') or {})
                    timeline_segments.append({
                        'abs_start': ev['start-time'],
                        'offset_seconds': (start_time - ref_time).total_seconds(),
                        'section': seg.get('timeline-section-title', ''),
                        'title': seg.get('timeline-segment-title', ''),
                    })
            except KeyError:
                continue

        timeline_segments.sort(key=lambda x: x['offset_seconds'])

        # Attendance
        attendance = []
        for cu in (data.get('class-users') or []):
            role = (cu.get('role') or '').lower()
            if role == 'student':
                u = cu.get('user') or {}
                first = u.get('first-name', '') or ''
                last = u.get('last-name', '') or ''
                name = f"{first} {last}".strip() or (u.get('preferred-name') or '').strip() or (u.get('first-name') or '').strip()
                uid = u.get('id') or u.get('user-id')
                absent = bool(cu.get('absent', False))
                attendance.append({'id': uid, 'name': name, 'absent': absent})

        try:
            attendance.sort(key=lambda x: (x['name'] or '').lower())
        except Exception:
            pass

        return {
            'class_id': class_id,
            'class_meta': class_meta,
            'voice_events': voice_events,
            'timeline_segments': timeline_segments,
            'attendance': attendance
        }


class AudioPreprocessor:
    @staticmethod
    def validate_and_fix_file(file_path):
        """Validates and preprocesses audio files"""
        if file_path.startswith('http'):
            # Download from URL
            return AudioPreprocessor._download_from_url(file_path)
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.lower().endswith('.mp4'):
                mp3_path = file_path.rsplit('.', 1)[0] + '.mp3'
                result = subprocess.run([
                    'ffmpeg', '-y', '-v', 'warning', '-xerror',
                    '-i', file_path, '-vn',
                    '-acodec', 'libmp3lame', '-ar', '44100', '-ab', '192k', '-f', 'mp3',
                    mp3_path
                ], capture_output=True, text=True, check=False)

                if result.returncode == 0 and Path(mp3_path).exists() and Path(mp3_path).stat().st_size > 0:
                    return AudioPreprocessor._convert_to_whisper_wav(mp3_path)
                else:
                    return AudioPreprocessor._python_extract_audio(file_path)

            elif file_path.lower().endswith(('.mp3', '.m4a', '.aac', '.ogg')):
                return AudioPreprocessor._convert_to_whisper_wav(file_path)

            elif file_path.lower().endswith('.wav'):
                return file_path

            else:
                raise ValueError(f"Unsupported file format: {file_path}")

        except Exception as e:
            raise RuntimeError(f"Error processing file: {str(e)}")
    
    @staticmethod
    def _download_from_url(url):
        """Download file from URL to temp location"""
        base = url.split('?', 1)[0]
        suffix = Path(base).suffix or ".mp4"
        local_name = tempfile.mktemp(suffix=suffix)
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return local_name
    
    @staticmethod
    def _convert_to_whisper_wav(audio_path):
        """Convert audio to WAV format optimized for Whisper"""
        wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_path,
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                wav_path
            ], capture_output=True, text=True, check=True)
            return wav_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert {audio_path} to WAV format")
    
    @staticmethod
    def _python_extract_audio(file_path):
        """Fallback audio extraction using PyDub"""
        wav_path = file_path.rsplit('.', 1)[0] + '_extracted.wav'
        try:
            audio = AudioSegment.from_file(file_path).set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(wav_path, format="wav")
            if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
                return wav_path
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}")


class TranscriptionProcessor:
    def __init__(self, segment_length=14400, model_name="medium", progress_callback=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.progress_callback = progress_callback
        
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)
            except Exception:
                pass

        self.model = whisper.load_model(model_name).to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()

        self.segment_length = int(segment_length)

    def transcribe(self, audio_path, class_id):
        """Transcribe audio and save as JSON"""
        try:
            audio = AudioSegment.from_file(audio_path)
            total_duration = len(audio) / 1000.0
            
            all_segments = []
            segment_times = range(0, int(total_duration), self.segment_length)
            
            for i, start_time in enumerate(segment_times):
                if self.progress_callback:
                    progress = (i / len(segment_times)) * 100
                    self.progress_callback(progress)
                
                remaining = total_duration - start_time
                duration = min(self.segment_length, remaining)

                start_ms = int(start_time * 1000)
                end_ms = int((start_time + duration) * 1000)
                segment = audio[start_ms:end_ms]

                temp_path = tempfile.mktemp(suffix='.wav')
                segment.export(temp_path, format="wav")

                try:
                    cast_ctx = torch.amp.autocast("cuda") if self.device == "cuda" else nullcontext()
                    with cast_ctx:
                        result = self.model.transcribe(
                            temp_path,
                            word_timestamps=True,
                            language="en",
                            task="transcribe",
                            fp16=(self.device == "cuda"),
                            condition_on_previous_text=True,
                            initial_prompt="This is a university lecture."
                        )

                        for seg in result.get("segments", []):
                            seg_start = float(seg.get("start", 0.0)) + start_time
                            seg_end = float(seg.get("end", 0.0)) + start_time

                            words = []
                            for w in seg.get("words", []) or []:
                                words.append({
                                    "word": str(w.get("word", "")).strip(),
                                    "start": float(w.get("start", 0.0)) + start_time,
                                    "end": float(w.get("end", 0.0)) + start_time
                                })

                            all_segments.append({
                                "start": seg_start,
                                "end": seg_end,
                                "text": self._normalize_sentence_spacing(str(seg.get("text", "")).strip()),
                                "words": words
                            })

                except Exception:
                    continue
                finally:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()

            if not all_segments:
                raise RuntimeError("No segments were successfully transcribed")

            transcript_path = tempfile.mktemp(suffix='.json')
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump({"segments": sorted(all_segments, key=lambda x: x["start"])}, f, indent=2)

            return transcript_path

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def _normalize_sentence_spacing(self, text):
        """Fix sentence spacing and cleanup text"""
        if not text:
            return text
        
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        text = text.replace('\u00A0', ' ')
        text = re.sub(r'\s*\n+\s*', ' ', text)
        text = re.sub(r'(\.\.\.)(?=\S)', r'\1 ', text)
        text = re.sub(r'(?<!\.)'r'([.!?])'r'(?=([""\'(\[]?[A-Za-z]))', r'\1 ', text)
        text = re.sub(r'([:;])(?=([""\'(\[]?[A-Za-z]))', r'\1 ', text)
        text = re.sub(r'([.!?][""\')\]])(?=\S)', r'\1 ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()


class OutputGenerator:
    def generate_outputs(self, class_id, headers, events_data, transcript_path, privacy_mode):
        """Generate PDF and CSV outputs"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Load transcript data
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Generate PDF
        pdf_path = self._generate_pdf(class_id, headers, events_data, transcript_data, privacy_mode)
        
        # Generate CSV
        csv_path = self._generate_csv(class_id, headers, events_data, transcript_data, privacy_mode)
        
        return pdf_path, csv_path
    
    def _generate_pdf(self, class_id, headers, events_data, transcript_data, privacy_mode):
        """Generate PDF transcript"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        class_meta = events_data.get('class_meta', {})
        timeline_segments = events_data.get('timeline_segments', [])
        attendance = events_data.get('attendance', [])
        student_ids_set = {a.get('id') for a in attendance if a.get('id') is not None}

        # Build speaker map
        speaker_map = {}
        for ev in events_data.get('voice_events', []):
            speaker_map[(ev['start'], ev['end'])] = ev.get('speaker', {})

        def find_speaker_at_time(t):
            for (start, end), spk in speaker_map.items():
                if start <= t <= end:
                    return self._label_from_actor(spk, privacy_mode, student_ids_set)
            return "Professor"

        # Combine consecutive segments by same speaker
        compiled_entries = []
        current = {'speaker': None, 'start_time': None, 'text': [], 'end_time': None}

        for seg in transcript_data['segments']:
            st, en, tx = seg['start'], seg['end'], seg['text'].strip()
            if not tx:
                continue
            spk = find_speaker_at_time(st)

            start_new = False
            if not current['speaker']:
                start_new = True
            elif current['speaker'] != spk:
                start_new = True
            elif current['end_time'] is not None and st - current['end_time'] > 2:
                start_new = True

            if start_new:
                if current['speaker']:
                    compiled_entries.append(current)
                current = {'speaker': spk, 'start_time': st, 'text': [tx], 'end_time': en}
            else:
                current['text'].append(tx)
                current['end_time'] = en

        if current['speaker']:
            compiled_entries.append(current)

        # Create PDF
        styles = getSampleStyleSheet()
        contribution_style = ParagraphStyle(
            'ContributionStyle', parent=styles['Normal'],
            fontName='Helvetica', fontSize=10, leading=12, wordWrap='CJK'
        )
        header_style = ParagraphStyle(
            'HeaderStyle', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=12, textColor=colors.whitesmoke, alignment=1
        )
        speaker_style = ParagraphStyle(
            'SpeakerStyle', parent=styles['Normal'],
            fontName='Helvetica', fontSize=10, leading=12, wordWrap='CJK'
        )

        output_path = tempfile.mktemp(suffix='.pdf')
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        elements = []

        # Header
        session_line = class_meta.get('session_title') or f"Session {class_id}"
        elements.append(Paragraph(session_line, styles['Title']))

        sec_sched = class_meta.get('section_title', '') or class_meta.get('schedule', '')
        if sec_sched:
            centered_info_style = ParagraphStyle('CenteredInfo', parent=styles['Heading3'], alignment=1)
            elements.append(Paragraph(sec_sched, centered_info_style))
            elements.append(Spacer(1, 12))
        else:
            elements.append(Spacer(1, 12))

        # Class info
        left_info_style = ParagraphStyle('LeftInfo', parent=styles['Normal'], alignment=0)
        class_datetime = self._fmt_dt_hm(class_meta.get('recording_start'))

        ref = (headers.get('referer') or headers.get('Referer') or '').strip()
        m = re.search(r'https://forum\.minerva\.edu/app/[^\s"\']+', ref)
        class_link = m.group(0) if m else f"https://forum.minerva.edu/app/classes/{class_id}"

        elements.append(Paragraph(f"<b>Class ID:</b> {class_id}", left_info_style))
        elements.append(Paragraph(f"<b>Class Date/Time:</b> {class_datetime}", left_info_style))
        elements.append(Paragraph(f'<b>Class Link:</b> <a href="{class_link}">{class_link}</a>', left_info_style))
        elements.append(Spacer(1, 12))

        # Attendance table
        if attendance:
            elements.append(Paragraph("Attendance", styles['Heading3']))
            att_rows = [[Paragraph('Student', header_style), Paragraph('Status', header_style)]]
            for a in attendance:
                status = 'Absent' if a.get('absent') else 'Present'
                display_student = (f"ID {a.get('id')}" if privacy_mode == 'ids' and a.get('id') is not None else a.get('name', ''))
                att_rows.append([Paragraph(self._soft_break_long_token(display_student, 14), speaker_style), status])
            att_table = Table(att_rows, colWidths=[4.5*inch, 1.5*inch], repeatRows=1)
            att_style = TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,0), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 10),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ])
            for i, a in enumerate(attendance, start=1):
                color = colors.red if a.get('absent') else colors.green
                att_style.add('TEXTCOLOR', (1,i), (1,i), color)
            att_table.setStyle(att_style)
            elements.append(att_table)
            elements.append(Spacer(1, 18))

        # Class events table
        if timeline_segments:
            elements.append(Paragraph("Class Events", styles['Heading3']))
            events_data_rows = [[Paragraph('Time', header_style),
                                Paragraph('Section', header_style),
                                Paragraph('Event', header_style)]]
            for seg in timeline_segments:
                sec_txt = self._soft_break_long_token(seg.get('section', '') or '', 14)
                evt_txt = self._soft_break_long_token(seg.get('title', '') or '', 14)
                events_data_rows.append([
                    self._fmt_mmss(seg.get('offset_seconds')),
                    Paragraph(sec_txt, contribution_style),
                    Paragraph(evt_txt, contribution_style)
                ])
            events_table = Table(events_data_rows, colWidths=[0.85*inch, 2.10*inch, 4.05*inch], repeatRows=1)
            events_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,0), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 10),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            elements.append(events_table)
            elements.append(Spacer(1, 18))

        # Transcript
        elements.append(Paragraph("Transcript", styles['Heading3']))
        elements.append(Spacer(1, 6))

        # Flatten entries to printable rows
        all_items = []
        for entry in compiled_entries:
            text = ' '.join(entry['text']).strip()
            if text in ['...', '.', '', 'Mm-hmm.'] or len(text) < 3:
                continue
            timestamp = self._fmt_mmss(entry['start_time'])

            max_chars_per_chunk = 500
            sentences = text.split('. ')
            chunks, curr = [], ""
            for s in sentences:
                candidate = (curr + s + '. ').strip() if curr else (s + '. ')
                if len(candidate) <= max_chars_per_chunk:
                    curr = candidate
                else:
                    if curr: chunks.append(curr.strip())
                    curr = s + '. '
            if curr: chunks.append(curr.strip())

            for i, chunk in enumerate(chunks or [text]):
                display_ts = "(cont.)" if i > 0 else timestamp
                all_items.append({
                    'start_time': entry['start_time'],
                    'end_time': entry['end_time'],
                    'timestamp': display_ts,
                    'speaker': entry['speaker'],
                    'text': chunk
                })

        all_items.sort(key=lambda x: x['start_time'])

        # Build event windows
        seg_windows = []
        if timeline_segments:
            first_start = max(0, (timeline_segments[0].get('offset_seconds') or 0))
            if first_start > 0:
                seg_windows.append({'start': 0, 'end': first_start, 'label': f"{self._fmt_mmss(0)} — Before first event"})
            for idx, seg in enumerate(timeline_segments):
                start = max(0, (seg.get('offset_seconds') or 0))
                end = (timeline_segments[idx+1].get('offset_seconds') if idx+1 < len(timeline_segments) else float('inf')) or float('inf')
                label_bits = []
                if seg.get('section'): label_bits.append(seg['section'])
                if seg.get('title'): label_bits.append(seg['title'])
                label_core = ' · '.join(label_bits) if label_bits else 'Event'
                seg_windows.append({'start': start, 'end': end, 'label': f"{self._fmt_mmss(start)} — {label_core}"})
        else:
            seg_windows.append({'start': 0, 'end': float('inf'), 'label': "Transcript"})

        for win in seg_windows:
            bucket = [it for it in all_items if win['start'] <= it['start_time'] < win['end']]
            if not bucket:
                continue

            elements.append(Paragraph(win['label'], styles['Heading4']))
            elements.append(Spacer(1, 4))

            data = [[Paragraph('Time', header_style),
                     Paragraph('Speaker', header_style),
                     Paragraph('Contribution', header_style)]]
            for item in bucket:
                spk_txt = self._soft_break_long_token(item['speaker'], 14)
                data.append([item['timestamp'], Paragraph(spk_txt, speaker_style), 
                           Paragraph(item['text'], contribution_style)])

            table = Table(data, colWidths=[0.75*inch, 2.2*inch, 4.25*inch], repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 16))

        doc.build(elements)
        return output_path
    
    def _generate_csv(self, class_id, headers, events_data, transcript_data, privacy_mode):
        """Generate CSV transcript"""
        class_meta = events_data.get('class_meta', {})
        timeline_segments = events_data.get('timeline_segments', [])
        attendance = events_data.get('attendance', [])
        student_ids_set = {a.get('id') for a in attendance if a.get('id') is not None}

        # Build speaker map
        speaker_map = {}
        for ev in events_data.get('voice_events', []):
            speaker_map[(ev['start'], ev['end'])] = ev.get('speaker', {})

        def find_speaker_at_time(t):
            for (start, end), spk in speaker_map.items():
                if start <= t <= end:
                    return self._label_from_actor(spk, privacy_mode, student_ids_set)
            return "Professor"

        # Combine segments by speaker
        compiled_entries = []
        current = {'speaker': None, 'start_time': None, 'text': [], 'end_time': None}
        for seg in transcript_data['segments']:
            st, en, tx = seg['start'], seg['end'], seg['text'].strip()
            if not tx:
                continue
            spk = find_speaker_at_time(st)

            start_new = False
            if not current['speaker']:
                start_new = True
            elif current['speaker'] != spk:
                start_new = True
            elif current['end_time'] is not None and st - current['end_time'] > 2:
                start_new = True

            if start_new:
                if current['speaker']:
                    compiled_entries.append(current)
                current = {'speaker': spk, 'start_time': st, 'text': [tx], 'end_time': en}
            else:
                current['text'].append(tx)
                current['end_time'] = en
        if current['speaker']:
            compiled_entries.append(current)

        # Flatten to rows
        all_items = []
        for entry in compiled_entries:
            text = ' '.join(entry['text']).strip()
            if text in ['...', '.', '', 'Mm-hmm.'] or len(text) < 3:
                continue
            timestamp = self._fmt_mmss(entry['start_time'])
            all_items.append({
                'timestamp': timestamp,
                'speaker': entry['speaker'],
                'text': text,
                'start_time': entry['start_time'],
                'end_time': entry['end_time']
            })
        all_items.sort(key=lambda x: x['start_time'])

        # Build event windows
        segmented_rows = []
        seg_windows = []
        if timeline_segments:
            first_start = max(0, (timeline_segments[0].get('offset_seconds') or 0))
            if first_start > 0:
                seg_windows.append({'start': 0, 'end': first_start, 'label': f"{self._fmt_mmss(0)} — Before first event"})
            for idx, seg in enumerate(timeline_segments):
                start = max(0, (seg.get('offset_seconds') or 0))
                end = (timeline_segments[idx+1].get('offset_seconds') if idx+1 < len(timeline_segments) else float('inf')) or float('inf')
                bits = []
                if seg.get('section'): bits.append(seg['section'])
                if seg.get('title'): bits.append(seg['title'])
                label = f"{self._fmt_mmss(start)} — " + (' / '.join(bits) if bits else 'Event')
                seg_windows.append({'start': start, 'end': end, 'label': label})
        else:
            seg_windows.append({'start': 0, 'end': float('inf'), 'label': "Transcript"})

        for win in seg_windows:
            bucket = [it for it in all_items if win['start'] <= it['start_time'] < win['end']]
            if not bucket:
                continue
            segmented_rows.append({'timestamp': '', 'speaker': '', 'text': f"--- {win['label']} ---"})
            segmented_rows.extend(bucket)

        all_items = segmented_rows or all_items

        # Write CSV
        output_path = tempfile.mktemp(suffix='.csv')
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            w = csv.writer(csvfile)

            # Header
            w.writerow(["Session", class_meta.get('session_title', '')])
            sec_sched = class_meta.get('section_title') or class_meta.get('schedule') or ''
            if sec_sched:
                w.writerow([sec_sched])
            w.writerow([])

            # Class info
            class_datetime = self._fmt_dt_hm(class_meta.get('recording_start'))
            ref = (headers.get('referer') or headers.get('Referer') or '').strip()
            m = re.search(r'https://forum\.minerva\.edu/app/[^\s"\']+', ref)
            class_link = m.group(0) if m else f"https://forum.minerva.edu/app/classes/{class_id}"

            w.writerow([f"Class ID: {class_id}"])
            w.writerow([f"Class Date/Time: {class_datetime}"])
            w.writerow([f"Class Link: {class_link}"])
            w.writerow([])

            # Attendance
            if attendance:
                w.writerow(["Attendance"])
                w.writerow(["Student", "Status"])
                for a in attendance:
                    label = (str(a['id']) if (privacy_mode == "ids" and a.get('id')) else a.get('name', ''))
                    w.writerow([label, "Absent" if a.get('absent') else "Present"])
                w.writerow([])

            # Class events
            if timeline_segments:
                w.writerow(["Class Events"])
                w.writerow(["Time", "Section", "Event"])
                for seg in timeline_segments:
                    w.writerow([
                        self._fmt_mmss(seg.get('offset_seconds')),
                        seg.get('section', ''),
                        seg.get('title', '')
                    ])
                w.writerow([])

            # Transcript table
            w.writerow(['Time', 'Speaker', 'Contribution'])
            for item in all_items:
                w.writerow([
                    item['timestamp'],
                    item['speaker'],
                    item['text']
                ])

        return output_path
    
    def _fmt_mmss(self, seconds_float):
        """Format seconds as MM:SS"""
        if seconds_float is None:
            return ""
        seconds = max(0, int(seconds_float))
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"
    
    def _fmt_dt_hm(self, dt_str):
        """Format datetime string"""
        if not dt_str:
            return ""
        try:
            dt = iso8601.parse_date(dt_str)
            return dt.strftime("%Y-%m-%d %H:%M %Z")
        except Exception:
            return dt_str.split('T')[0] if 'T' in dt_str else ""
    
    def _soft_break_long_token(self, s, max_run=14):
        """Add soft breaks to long tokens"""
        if not s:
            return s
        pat = re.compile(r'(\S{%d})(?=\S)' % max_run)
        return pat.sub(lambda m: m.group(1) + '\u200b', s)
    
    def _label_from_actor(self, actor, name_mode, student_ids=None):
        """Return display label for a speaker based on privacy setting"""
        if not isinstance(actor, dict):
            return "Professor"
        uid = actor.get('id') or actor.get('user-id') or (actor.get('user') or {}).get('id')
        fn = (actor.get('first-name') or '').strip()
        ln = (actor.get('last-name') or '').strip()
        full = f"{fn} {ln}".strip()
        if name_mode == 'ids':
            if (student_ids is None and uid is not None) or (student_ids is not None and uid in student_ids):
                return str(uid) if uid is not None else "ID"
        return full or "Professor"