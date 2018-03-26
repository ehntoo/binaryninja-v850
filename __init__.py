from __future__ import print_function

import struct, ctypes

from collections import defaultdict
from binaryninja import (
    Architecture, RegisterInfo, InstructionInfo,
    InstructionTextToken, InstructionTextTokenType,
    BranchType,
    LowLevelILOperation, LLIL_TEMP,
    LowLevelILLabel,
    FlagRole,
    LowLevelILFlagCondition,
    log_error,
    CallingConvention)

def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)

REGISTER_REGISTER_MODE = 0         # r1, r2
SINGLE_REGISTER_MODE = 1           # r1
INDIRECT_REGISTER_MODE = 2         # [r1]
INDIRECT_REGISTER_OFFSET_MODE = 3  # 
ELEMENT_OFFSET_LOAD_MODE = 4       # sld.bu -25 [ep], r2
ELEMENT_OFFSET_STORE_MODE = 5      # sst.b r2, -25 [ep]
IMMEDIATE_REGISTER_MODE = 6        # mov 5, r2
IMMEDIATE_MODE = 7                 # callt 25
OFFSET_MODE = 8                    # jr 123
OFFSET_REGISTER_MODE = 9           # 

Registers = [
    'r0',
    'r1',
    'r2',
    'sp',
    'r4',
    'r5',
    'r6',
    'r7',
    'r8',
    'r9',
    'r10',
    'r11',
    'r12',
    'r13',
    'r14',
    'r15',
    'r16',
    'r17',
    'r18',
    'r19',
    'r20',
    'r21',
    'r22',
    'r23',
    'r24',
    'r25',
    'r26',
    'r27',
    'r28',
    'r29',
    'ep',
    'lp',
    'pc',
]

NoRegisterInstructions = {
    0x0: 'nop',
    0x1d: 'synce',
    0x1e: 'syncm',
    0x1f: 'syncp',
    0x40: 'rie',
}

OneRegInstructions = [
    'switch', 'zxb', 'sxb', 'zxh', 'sxh'
]

TwoRegInstructions = [
    'mov', 'not', 'divh', 'satsubr', 'satsub', 'satadd', 'mulh', 'or', 'xor',
    'and', 'tst', 'subr', 'sub', 'add', 'cmp',
]

RegImmediateInstructions = [
    'mov', 'satadd', 'add', 'cmp', 'shr', 'sar', 'shl', 'mulh'
]

TypeOneRegRegInstructions = {
    0x00: lambda reg1, reg2: ('mov', reg1, reg2),
    0x01: lambda reg1, reg2: ('not', reg1, reg2),
    0x02: lambda reg1, reg2: ('switch', None, reg1) if reg2 == 0 else ('fetrap', None, reg2 & 0xf) if reg1 == 0 else ('divh', reg1, reg2),
    0x04: lambda reg1, reg2: ('satsubr', reg1, reg2) if reg2 == 0 else ('zxb', None, reg1),
    0x05: lambda reg1, reg2: ('satsub', reg1, reg2) if reg2 == 0 else ('sxb', None, reg1),
    0x06: lambda reg1, reg2: ('satadd', reg1, reg2) if reg2 == 0 else ('zxh', None, reg1),
    0x07: lambda reg1, reg2: ('mulh', reg1, reg2) if reg2 == 0 else ('sxh', None, reg1),
    0x08: lambda reg1, reg2: ('or', reg1, reg2),
    0x09: lambda reg1, reg2: ('xor', reg1, reg2),
    0x0a: lambda reg1, reg2: ('and', reg1, reg2),
    0x0b: lambda reg1, reg2: ('tst', reg1, reg2),
    0x0c: lambda reg1, reg2: ('subr', reg1, reg2),
    0x0d: lambda reg1, reg2: ('sub', reg1, reg2),
    0x0e: lambda reg1, reg2: ('add', reg1, reg2),
    0x0f: lambda reg1, reg2: ('cmp', reg1, reg2),
}

TypeOneImmediateRegInstructions = {
    0x03: lambda reg1, reg2: ('jmp', reg1, None) if reg2 == 0 else ('sld.hu', (reg1 & 0xf) << 1, reg2) if reg1 & 0x10 == 0x10 else ('sld.bu', reg1 & 0xf, reg2),
    0x10: lambda reg1, reg2: ('mov', sign_extend(reg1, 5), reg2),
    0x11: lambda reg1, reg2: ('satadd', reg1, reg2),
    0x12: lambda reg1, reg2: ('add', sign_extend(reg1, 5), reg2),
    0x13: lambda reg1, reg2: ('cmp', sign_extend(reg1, 5), reg2),
    0x14: lambda reg1, reg2: ('callt', reg2 << 1, None) if reg1 == 0 else ('shr', reg1, reg2),
    0x15: lambda reg1, reg2: ('callt', (reg2 | 0x20) << 1, None) if reg1 == 0 else ('sar', reg1, reg2),
    0x16: lambda reg1, reg2: ('shl', reg1, reg2),
}

TypeFourShortLoadStoreInstructions = {
    0x6: lambda reg2, disp: ('sld.b', None, reg2, disp),
    0x7: lambda reg2, disp: ('sst.b', reg2, None, disp),
    0x8: lambda reg2, disp: ('sld.h', None, reg2, disp << 1),
    0x9: lambda reg2, disp: ('sst.h', reg2, None, disp << 1),
    0xa: lambda reg2, disp: ('sld.w', None, reg2, disp << 1) if disp & 0x1 == 0 else ('sst.w', reg2, None, (disp & 0x7e) << 1),
}

BitManipulationInstructions = [
    lambda reg1, bitnum, disp: ('set1', 4, reg1, bitnum, sign_extend(disp, 16)),
    lambda reg1, bitnum, disp: ('not1', 4, reg1, bitnum, sign_extend(disp, 16)),
    lambda reg1, bitnum, disp: ('clr1', 4, reg1, bitnum, sign_extend(disp, 16)),
    lambda reg1, bitnum, disp: ('tst1', 4, reg1, bitnum, sign_extend(disp, 16)),
]

class Register_List_Bits (ctypes.LittleEndianStructure):
    _fields_ = [
        ('r31', ctypes.c_uint16, 1),
        ('r29', ctypes.c_uint16, 1),
        ('r28', ctypes.c_uint16, 1),
        ('r23', ctypes.c_uint16, 1),
        ('r22', ctypes.c_uint16, 1),
        ('r21', ctypes.c_uint16, 1),
        ('r20', ctypes.c_uint16, 1),
        ('r26', ctypes.c_uint16, 1),
        ('r25', ctypes.c_uint16, 1),
        ('r24', ctypes.c_uint16, 1),
        ('r30', ctypes.c_uint16, 1),
    ]
class Register_List (ctypes.Union):
    _anonymous_ = ('bit',)
    _fields_ = [
        ('bit', Register_List_Bits),
        ('asWord', ctypes.c_uint16)
    ]

def decode_dispose(i1, i2):
    immed = ((i1 >> 1) & 0x1f) << 2
    reg1 = i2 & 0x1f
    reg_list = Register_List()
    reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)

    return 'dispose', 4, reg1 if reg1 != 0 else None, reg_list, immed

def decode_3c_and_3d(data):
    i1 = struct.unpack('<H', data[0:2])[0]
    i2 = struct.unpack('<H', data[2:4])[0]
    reg2 = (i1 >> 11) & 0x1f

    if i2 & 0x1 == 0 and reg2 != 0:
        return 'jarl', 4, None, reg2, sign_extend(((i1 & 0x3f) << 16) | i2, 22)
    if i2 & 0x1 == 0 and reg2 == 0:
        return 'jr', 4, None, None, sign_extend(((i1 & 0x3f) << 16) | i2, 22)
    if reg2 != 0:
        reg1 = (i1 & 0x1f)
        low_bit = (i1 >> 5) & 0x1
        return 'ld.bu', 4, reg1, reg2, sign_extend((i2 & 0xfffe) | low_bit, 16)

    subop = i2 & 0x1f

    if subop == 1:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        return 'prepare', 4, None, reg_list, immed
    if subop == 3:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        return 'prepare', 4, Registers.index('sp'), reg_list, immed
    if subop == 0xb or subop == 0x13:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        i3 = struct.unpack('<H', data[4:6])[0]
        return 'prepare', 6, sign_extend(i3, 16) if subop == 0xb else i3 << 16, reg_list, immed
    if subop == 0x1b:
        immed = ((i1 >> 1) & 0x1f) << 2
        reg_list = Register_List()
        reg_list.asWord = ((i1 & 0x1) << 11) | (i2 >> 5)
        i3 = struct.unpack('<H', data[4:6])[0]
        i4 = struct.unpack('<H', data[6:8])[0]
        return 'prepare', 8, i4 << 16 | i3, reg_list, immed

    composite_opcode = (i1 & 0x20) | (i2 & 0x1f)
    reg3 = (i2 >> 11) & 0x1f
    return LongLoadStoreInstructions[composite_opcode](reg1, reg3, disp22)

LongLoadStoreInstructions = {
    0x05: lambda reg1, reg3, disp22: ('ld.b', 6, reg1, reg3, sign_extend(disp22 << 1), 23),
    0x15: lambda reg1, reg3, disp22: ('ld.b', 6, reg1, reg3, sign_extend((disp22 << 1) | 1), 23),
    0x07: lambda reg1, reg3, disp22: ('ld.h', 6, reg1, reg3, sign_extend(disp22 << 1), 23),
    0x09: lambda reg1, reg3, disp22: ('ld.w', 6, reg1, reg3, sign_extend(disp22 << 1), 23),
    0x0d: lambda reg1, reg3, disp22: ('st.b', 6, reg3, reg1, sign_extend(disp22 << 1), 23),
    0x1d: lambda reg1, reg3, disp22: ('st.b', 6, reg3, reg1, sign_extend((disp22 << 1) | 1), 23),
    0x0f: lambda reg1, reg3, disp22: ('st.w', 6, reg3, reg1, sign_extend(disp22 << 1), 23),
    0x25: lambda reg1, reg3, disp22: ('ld.bu', 6, reg1, reg3, sign_extend(disp22 << 1), 23),
    0x35: lambda reg1, reg3, disp22: ('ld.bu', 6, reg1, reg3, sign_extend((disp22 << 1) | 1), 23),
    0x27: lambda reg1, reg3, disp22: ('ld.hu', 6, reg1, reg3, sign_extend(disp22 << 1), 23),
    0x2d: lambda reg3, reg1, disp22: ('st.h', 6, reg1, reg3, sign_extend(disp22 << 1), 23),
}

SimpleThirtyTwoBitInstructions = {
    0x30: lambda reg1, reg2, _, i2, _2: ('addi', 4, reg1, reg2, sign_extend(i2, 16)),
    0x31: lambda reg1, reg2, _, i2, i3: ('movea', 4, reg1, reg2, sign_extend(i2, 16)) if reg2 != 0 else ('mov', 6, None, reg1, (i3 << 16) | i2),
    0x32: lambda reg1, reg2, i1, i2, i3: ('movhi', 4, reg1, reg2, i2 << 16) if reg2 != 0 else decode_dispose(i1, i2),
    0x33: lambda reg1, reg2, i1, i2, i3: ('satsubi', 4, reg1, reg2, i2 << 16) if reg2 != 0 else decode_dispose(i1, i2),
    0x34: lambda reg1, reg2, _, i2, _2: ('ori', 4, reg1, reg2, i2),
    0x35: lambda reg1, reg2, _, i2, _2: ('xori', 4, reg1, reg2, i2),
    0x36: lambda reg1, reg2, _, i2, _2: ('andi', 4, reg1, reg2, i2),
    0x37: lambda reg1, reg2, _, i2, i3: ('mulhi', 4, reg1, reg2, i2) if reg2 != 0 else ('jmp', 6, reg1, None, (i2 << 16) | i3),
    0x38: lambda reg1, reg2, _, i2, _2: ('ld.b', 4, reg1, reg2, sign_extend(i2, 16)),
    0x39: lambda reg1, reg2, _, i2, _2: ('ld.h', 4, reg1, reg2, sign_extend(i2, 16)) if i2 & 0x1 == 0 else ('ld.w', 4, reg1, reg2, sign_extend(i2 & 0xfffe, 16)),
    0x3a: lambda reg1, reg2, _, i2, _2: ('st.b', 4, reg2, reg1, sign_extend(i2, 16)),
    0x3b: lambda reg1, reg2, _, i2, _2: ('st.h', 4, reg2, reg1, sign_extend(i2, 16)) if i2 & 0x1 == 0 else ('st.w', 4, reg2, reg1, sign_extend(i2 & 0xfffe, 16)),
}

ConditionCode = [
    'v', 'l', 'z', 'nh',
    'n', 't', 'lt', 'le',
    'nv', 'nl', 'nz', 'h',
    'p', 'sa', 'ge', 'gt'
]


class V850(Architecture):
    name = 'V850'
    address_size = 4
    default_int_size = 4
    max_instr_length = 8
    stack_pointer = 'r3'
    link_reg = 'lp'

    regs = {
        'r0': RegisterInfo('r0', 4),
        'r1': RegisterInfo('r1', 4),
        'r2': RegisterInfo('r2', 4),
        'r3': RegisterInfo('r3', 4),
        'r4': RegisterInfo('r4', 4),
        'r5': RegisterInfo('r5', 4),
        'r6': RegisterInfo('r6', 4),
        'r7': RegisterInfo('r7', 4),
        'r8': RegisterInfo('r8', 4),
        'r9': RegisterInfo('r9', 4),
        'r10': RegisterInfo('r10', 4),
        'r11': RegisterInfo('r11', 4),
        'r12': RegisterInfo('r12', 4),
        'r13': RegisterInfo('r13', 4),
        'r14': RegisterInfo('r14', 4),
        'r15': RegisterInfo('r15', 4),
        'r16': RegisterInfo('r16', 4),
        'r17': RegisterInfo('r17', 4),
        'r18': RegisterInfo('r18', 4),
        'r19': RegisterInfo('r19', 4),
        'r20': RegisterInfo('r20', 4),
        'r21': RegisterInfo('r21', 4),
        'r22': RegisterInfo('r22', 4),
        'r23': RegisterInfo('r23', 4),
        'r24': RegisterInfo('r24', 4),
        'r25': RegisterInfo('r25', 4),
        'r26': RegisterInfo('r26', 4),
        'r27': RegisterInfo('r27', 4),
        'r28': RegisterInfo('r28', 4),
        'r29': RegisterInfo('r29', 4),
        'ep': RegisterInfo('ep', 4),
        'lp': RegisterInfo('lp', 4),
        'pc': RegisterInfo('pc', 4),
    }

    flags = ['sat', 'cy', 'ov', 's', 'z']
    flag_roles = {
        'sat': FlagRole.SpecialFlagRole,
        'cy': FlagRole.CarryFlagRole,
        'ov': FlagRole.OverflowFlagRole,
        's': FlagRole.NegativeSignFlagRole,
        'z': FlagRole.ZeroFlagRole,
    }
    flags_required_for_flag_condition = {
        LowLevelILFlagCondition.LLFC_UGE: ['cy'],
        LowLevelILFlagCondition.LLFC_ULT: ['cy'],
        LowLevelILFlagCondition.LLFC_SGE: ['s', 'ov'],
        LowLevelILFlagCondition.LLFC_SLT: ['s', 'ov'],
        LowLevelILFlagCondition.LLFC_E: ['z'],
        LowLevelILFlagCondition.LLFC_NE: ['z'],
        LowLevelILFlagCondition.LLFC_NEG: ['s'],
        LowLevelILFlagCondition.LLFC_POS: ['s']
    }

    # The first flag write type is ignored currently.
    # See: https://github.com/Vector35/binaryninja-api/issues/513
    # flag_write_types = ['', '*', 'cnv', 'cnz']
    flag_write_types = ['*']
    flags_written_by_flag_write_type = {
        "*": ["cy", "z", "ov", "s"],
    }

    def decode_instruction(self, data, addr):
        error_value = (None, None, None, None, None)
        if len(data) < 2:
            return error_value

        instruction = struct.unpack('<H', data[0:2])[0]
        # is this a zero-register instruction?
        if instruction in NoRegisterInstructions:
            return (NoRegisterInstructions[instruction], 2, None, None, None)

        opcode = (instruction >> 5) & 0x3f
        reg1 = (instruction & 0x1f)
        reg2 = (instruction >> 11) & 0x1f

        if opcode in TypeOneRegRegInstructions:
            instr, src, dst = TypeOneRegRegInstructions[opcode](reg1, reg2)
            return (instr, 2, src, dst, None)

        if opcode in TypeOneImmediateRegInstructions:
            instr, immed, dst = TypeOneImmediateRegInstructions[opcode](reg1, reg2)
            return (instr, 2, None, dst, immed)

        # 0x17: mulh, jr, jarl
        if opcode == 0x17:
            if reg1 == reg2 == 0:
                return 'jr', 6, None, None, (struct.unpack('<H', data[4:6])[0] << 16) | struct.unpack('<H', data[2:4])[0]
            if reg2 == 0:
                return 'jarl', 6, None, reg1, sign_extend((struct.unpack('<H', data[4:6])[0] << 16) | struct.unpack('<H', data[2:4])[0], 32) 
            else:
                return 'mulh', 2, None, reg2, reg1

        # Handle short load store instructions
        if opcode >= 0x18 and opcode <= 0x2b:
            disp = instruction & 0x7f
            load_store_type = opcode >> 2
            instr, src, dst, immed = TypeFourShortLoadStoreInstructions[load_store_type](reg2, disp)
            return instr, 2, src, dst, immed

        # Handle branches
        if opcode >= 0x2c and opcode <= 0x2f:
            disp = (reg2 << 3) | ((instruction >> 4) & 0x7)
            return 'b'+ConditionCode[instruction & 0xf], 2, None, None, sign_extend(disp, 9)

        instruction2 = struct.unpack('<H', data[2:4])[0]
        instruction3 = struct.unpack('<H', data[4:6])[0]

        if opcode in SimpleThirtyTwoBitInstructions:
            return SimpleThirtyTwoBitInstructions[opcode](reg1, reg2, instruction, instruction2, instruction3)

        if opcode == 0x3c or opcode == 0x3d:
            return decode_3c_and_3d(data)
        if opcode == 0x3e:
            bitop = instruction >> 14
            bitnum = (instruction >> 11) & 0x7
            return BitManipulationInstructions[bitop](reg1, bitnum, instruction2)

        # return instr, length, src_reg, dst_reg, immed
        return error_value

    def perform_get_instruction_info(self, data, addr):
        instr, length, src_reg, _, immed = self.decode_instruction(data, addr)
        if instr is None:
            return None

        result = InstructionInfo()
        result.length = length

        return result
        # instr, _, _, _, _, _, length, src_value, _ = self.decode_instruction(data, addr)
        #
        # if instr is None:
        #     return None
        #
        # result = InstructionInfo()
        # result.length = length
        #
        # # Add branches
        # if instr in ['ret', 'reti']:
        #     result.add_branch(BranchType.FunctionReturn)
        # elif instr in ['jmp', 'br'] and src_value is not None:
        #     result.add_branch(BranchType.UnconditionalBranch, src_value)
        # elif instr in TYPE3_INSTRUCTIONS:
        #     result.add_branch(BranchType.TrueBranch, src_value)
        #     result.add_branch(BranchType.FalseBranch, addr + 2)
        # elif instr == 'call' and src_value is not None:
        #     result.add_branch(BranchType.CallDestination, src_value)
        #
        # return result

    def perform_get_instruction_text(self, data, addr):
        instr, length, src_reg, dst_reg, immed = self.decode_instruction(data, addr)
        if instr is None:
            return None

        instruction_text = instr
        tokens = [
            InstructionTextToken(InstructionTextTokenType.TextToken, '{:7s}'.format(instruction_text))
        ]

        # if instr in NoRegisterInstructions.values():
        #     return tokens, length
        if instr in OneRegInstructions and dst_reg is not None:
            tokens += InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])

        if instr in TwoRegInstructions and src_reg is not None and dst_reg is not None and immed is None:
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[src_reg])]
            tokens += [InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', ')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr in RegImmediateInstructions and src_reg is None and dst_reg is not None and immed is not None:
            # print("instr: ", instr, "immediate: ", immed, "register: ", Registers[dst_reg])
            tokens += [InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(immed))]
            tokens += [InstructionTextToken(InstructionTextTokenType.TextToken, ',')]
            tokens += [InstructionTextToken(InstructionTextTokenType.RegisterToken, Registers[dst_reg])]

        if instr[0] == 'b':
            # TODO - fix targets
            tokens += [InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(immed))]
            # tokens += [InstructionTextToken(InstructionTextTokenType.TextToken, immed)]

        return tokens, length
        # (instr, width,
        #  src_operand, dst_operand,
        #  src, dst, length,
        #  src_value, dst_value) = self.decode_instruction(data, addr)
        #
        # if instr is None:
        #     return None
        #
        # tokens = []
        #
        # instruction_text = instr
        #
        # if width == 1:
        #     instruction_text += '.b'
        #
        # tokens = [
        #     InstructionTextToken(InstructionTextTokenType.TextToken, '{:7s}'.format(instruction_text))
        # ]
        #
        # if instr in TYPE1_INSTRUCTIONS:
        #     tokens += OperandTokens[src_operand](src, src_value)
        #
        #     tokens += [InstructionTextToken(InstructionTextTokenType.TextToken, ',')]
        #
        #     tokens += OperandTokens[dst_operand](dst, dst_value)
        #
        # elif instr in TYPE2_INSTRUCTIONS:
        #     tokens += OperandTokens[src_operand](src, src_value)
        #
        # elif instr in TYPE3_INSTRUCTIONS:
        #     tokens += OperandTokens[src_operand](src, src_value)
        #
        # return tokens, length

    # def perform_get_instruction_low_level_il(self, data, addr, il):
        #     il.append(il.unimplemented())
        # (instr, width,
        #     src_operand, dst_operand,
        #     src, dst, length,
        #     src_value, dst_value) = self.decode_instruction(data, addr)
        #
        # if instr is None:
        #     return None
        #
        # if InstructionIL.get(instr) is None:
        #     log_error('[0x{:4x}]: {} not implemented'.format(addr, instr))
        #     il.append(il.unimplemented())
        # else:
        #     il_instr = InstructionIL[instr](
        #         il, src_operand, dst_operand, src, dst, width, src_value, dst_value
        #     )
        #     if isinstance(il_instr, list):
        #         for i in [i for i in il_instr if i is not None]:
        #             il.append(i)
        #     elif il_instr is not None:
        #         il.append(il_instr)

        # return length

# class DefaultCallingConvention(CallingConvention):
#     int_arg_regs = ['r15', 'r14', 'r13', 'r12']
#     int_return_reg = 'r15'
#     high_int_return_reg = 'r14'

V850.register()
arch = Architecture['V850']
# arch.register_calling_convention(DefaultCallingConvention(arch, 'default'))
# standalone = arch.standalone_platform
# standalone.default_calling_convention = arch.calling_conventions['default']
